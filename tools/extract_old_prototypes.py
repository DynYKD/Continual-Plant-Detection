
import argparse
from symbol import return_stmt
import torch

from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size
from maskrcnn_benchmark.config import cfg

from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data import make_data_loader 
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.solver import make_lr_scheduler  # learning rate updating strategy
from maskrcnn_benchmark.solver import make_optimizer  
import time
from maskrcnn_benchmark.modeling.utils import cat
import numpy as np
from tqdm import tqdm
import pickle

def get_data_loader(cfg):
    #Dumb fix to automate cfg.SOLVER.MAX_ITER
    cfg.SOLVER.MAX_ITER = 7779 // cfg.SOLVER.IMS_PER_BATCH  
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,  # whether using multiple gpus to train
        start_iter=0,
        num_gpus=get_world_size(),
        rank=get_rank(),
        remove_old_classes=False
    )
    cfg.SOLVER.MAX_ITER = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH  
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,  # whether using multiple gpus to train
        start_iter=0,
        num_gpus=get_world_size(),
        rank=get_rank(),
        remove_old_classes=False
    )
    return data_loader

class NEW_BatchPrototypesExtractor:
    def __init__(self, labels_pos, mask_targets, positive_inds):
        self.labels_pos, self.mask_targets, self.positive_inds = labels_pos, mask_targets, positive_inds
        #self.n_old_classes = n_old_classes
        #self.n_all_classes = n_all_classes
    def __call__(self, target_mask_logits):
        n_classes = target_mask_logits.shape[1]
        feature_map_size = target_mask_logits.shape[-1]
        target_mask_logits = target_mask_logits.permute(0,2,3,1)
        new_protos = self.extract_proto_features(target_mask_logits)[:,1:]
        #new_protos = self.build_new_protos(new_proto_features)[:,1:16]
        
        prototype_map = torch.zeros((len(self.positive_inds), feature_map_size, feature_map_size, n_classes-1)).cuda()
        labels_idx = torch.unique(self.labels_pos)
        
        for i in range(len(self.positive_inds)):
            prototype_map[i, self.mask_targets[i].bool()] = new_protos[torch.where(labels_idx==self.labels_pos[i])]
        prototype_map = torch.nn.functional.normalize(prototype_map, dim=-1)
        
        return prototype_map

    def build_new_protos(self, new_proto_features):
        new_protos = []
        for i in range(len(new_proto_features)):
            new_proto = np.mean(np.concatenate(new_proto_features[i]), axis=0)
            new_protos.append(new_proto)
        new_protos = torch.Tensor(np.array(new_protos))
        return new_protos

    def extract_proto_features(self, target_mask_logits):                            
        labels = []          
        n_classes = target_mask_logits.shape[-1]
        labels_idx = torch.unique(self.labels_pos)
        new_proto_features = torch.zeros((len(labels_idx), n_classes)).cuda()
        
        for i in range(len(self.positive_inds)):
            labels.append(self.labels_pos[i])
            new_proto_features[torch.where(labels_idx==self.labels_pos[i])] += torch.sum(target_mask_logits[self.positive_inds[i]][self.mask_targets[i].bool()], dim=0)
        return new_proto_features

class BatchPrototypesExtractor:
    def __init__(self, labels_pos, mask_targets, positive_inds):
        self.labels_pos, self.mask_targets, self.positive_inds = labels_pos, mask_targets, positive_inds

    def __call__(self, target_mask_logits):
        target_mask_logits = target_mask_logits.permute(0,2,3,1)
        B, H, W , C = target_mask_logits.shape
        new_proto_features = self.extract_proto_features(target_mask_logits)
        new_protos = self.build_new_protos(new_proto_features)
        return new_protos

    def build_new_protos(self, new_proto_features):
        new_protos = []
        for i in range(len(new_proto_features)):
            new_proto = np.sum(np.concatenate(new_proto_features[i]), axis=0)
            new_protos.append(new_proto)
        new_protos = torch.Tensor(np.array(new_protos))
        return new_protos

    def extract_proto_features(self, target_mask_logits):
        new_proto_features = []
        for i in range(len(self.positive_inds)):
            new_proto_features.append([self.labels_pos[i].cpu().numpy(), target_mask_logits[self.positive_inds[i]][self.mask_targets[i].bool()].clone().detach().cpu().numpy()])
        new_proto_features = np.array(new_proto_features)
        #import pdb
        #pdb.set_trace()
        #new_proto_features[:, 0].argsort()
        if len(new_proto_features) > 0:
            new_proto_features = new_proto_features[new_proto_features[:, 0].argsort()]
            new_proto_features = np.split(new_proto_features[:,1], np.unique(new_proto_features[:, 0], return_index=True)[1][1:])
        return new_proto_features

def get_pos_preds(model, soften_proposal, targets):
    labels, mask_targets = model.roi_heads.mask.loss_evaluator.prepare_targets(soften_proposal, targets)
    labels = cat(labels, dim=0)
    mask_targets = cat(mask_targets, dim=0)
    positive_inds = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[positive_inds]
    return labels_pos, mask_targets, positive_inds

def get_pos_preds_classif(model, soften_proposal, targets):
    labels, _ = model.roi_heads.box.loss_evaluator.prepare_targets(soften_proposal, targets)
    labels = cat(labels, dim=0)
    positive_inds = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[positive_inds]
    mask_targets = torch.ones_like(labels_pos).unsqueeze(-1).unsqueeze(-1)
    return labels_pos, mask_targets, positive_inds

def extract_segmentation_prototypes(model, soften_proposal, soften_mask_logits, targets):
    labels_pos, mask_targets, positive_inds = get_pos_preds(model, soften_proposal, targets)
    labels_idx, labels_counts = np.unique(labels_pos.cpu().numpy(), return_counts=True)
    batch_prototype_extractor = BatchPrototypesExtractor(labels_pos, mask_targets, positive_inds)
    new_protos = batch_prototype_extractor(soften_mask_logits)
    return new_protos, labels_idx, labels_counts

def extract_classif_prototypes(model, soften_proposal, soften_result, targets):
    labels_pos, mask_targets, positive_inds = get_pos_preds_classif(model, soften_proposal, targets)
    labels_idx, labels_counts = np.unique(labels_pos.cpu().numpy(), return_counts=True)
    batch_prototype_extractor = BatchPrototypesExtractor(labels_pos, mask_targets, positive_inds)
    new_protos = batch_prototype_extractor(soften_result[0].unsqueeze(-1).unsqueeze(-1))
    return new_protos, labels_idx, labels_counts

def store_segmentation_prototypes(new_protos, old_prototypes, n_old_proto, labels_idx, labels_counts):
    for i, proto in enumerate(new_protos):
        old_prototypes[labels_idx[i]-1] += proto[1:]
        n_old_proto[labels_idx[i]-1] += labels_counts[i]
    return old_prototypes, n_old_proto

def store_classif_prototypes(new_protos, old_prototypes, n_old_proto, labels_idx, labels_counts):
    for i, proto in enumerate(new_protos):
        old_prototypes[labels_idx[i]-1] += proto[1:]
        n_old_proto[labels_idx[i]-1] += labels_counts[i]
    return old_prototypes, n_old_proto

def mask_average_pooling(targets, features):    
    all_labels = []
    all_masked_avg_features = []
    all_binary_masks = []
    for i in range(len(targets)):
        t = targets[i]
        f = features[i]
        binary_masks = t.get_field("masks").instances.masks
        t_downsample = torch.nn.functional.interpolate(binary_masks.unsqueeze(1), features.shape[2:]).squeeze(1)
        all_binary_masks.append(binary_masks)
        for j in range(len(t_downsample)):
            mask_instance = t_downsample[j]
            n_features = torch.sum(mask_instance).cpu().numpy()
            if n_features > 0:
                random_feature = torch.tensor(np.random.randint(0, n_features)).cuda()
                masked_avg_features = f[:, mask_instance][:,random_feature]# torch.sum(f * mask_instance.unsqueeze(0), dim=(1,2)) / torch.sum(mask_instance)
                all_labels.append(t.get_field("labels")[j].cpu().numpy())
                all_masked_avg_features.append(masked_avg_features)
    return all_labels, all_masked_avg_features, all_binary_masks


def crop_features(target, features):
    
    all_labels = []
    all_cropped_features = []
    all_binary_masks = []
    f = features
    binary_masks = target.get_field("masks").instances.masks
    mask_instances = torch.nn.functional.interpolate(binary_masks.unsqueeze(1), features.shape[2:]).squeeze(1).squeeze(1)
    all_binary_masks.append(binary_masks)

    for j in range(len(binary_masks)):
        cropped_features = f * mask_instances[j].unsqueeze(0)
        all_cropped_features.append(cropped_features)
        all_labels.append(target.get_field("labels")[j].cpu().numpy())

    return all_cropped_features                

def sample_features(n_per_class, features, labels, binary_masks):
    sampled_features = []
    sampled_binary_masks = []
    for c in np.unique(labels):
        idx = np.arange(len(labels))
        sampled_idx_c = np.random.choice(idx[labels == c], n_per_class, replace=False)
        sampled_features_c = features[sampled_idx_c]
        sampled_masks_c = []
        sampled_features.append(sampled_features_c)
        counter = 0
        for i in range(len(binary_masks)):
            for j in range(len(binary_masks[i])):
                for k in range(len(binary_masks[i][j])):
                    if counter in sampled_idx_c:
                        sampled_masks_c.append(binary_masks[i][j][k].cpu())
                    counter += 1
            
        sampled_binary_masks.append(sampled_masks_c)

    sampled_features = np.array(sampled_features)
    #sampled_binary_masks = np.array(sampled_binary_masks)
    return sampled_features, sampled_binary_masks

def filter_examplars(images, batch_features, target, model):
    from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

    # Conditions:
    # mask is not too big nor too small
    # IoU > 0.90
    # predicted class is the good one
    target_masks = target.get_field("masks").instances.masks
    target_boxes = target.bbox
    target.get_field("labels")
    target_masks.shape
    
    with torch.no_grad(): preds = model.forward(images)
    ious = boxlist_iou(preds[0][0], target)
    if len(preds[0][0])>0:
        for i in range(ious.shape[1]):
            size = torch.sum(target_masks[i])
            if size < (300*300) and size > (100*100):                                  # mask is not too big nor too small
                iou_max = torch.max(ious[:,i])
                if iou_max > 0.9:                                                       # IoU > 0.90
                    best_pred_idx = torch.where(ious[:,i]==iou_max)[0]
                    predicted_class = preds[0][0].get_field("labels")[best_pred_idx]
                    if predicted_class == target.get_field("labels")[i]: # predicted class is the good one
                        kept_examplar = (target.get_field("labels")[i], batch_features[i], target_masks[i])
                        return kept_examplar

def get_memory_rois(model, soften_proposal, targets, soften_result, roi_align_features):
    labels_pos, mask_targets, positive_inds = get_pos_preds(model, soften_proposal, targets)
    correct_preds_inds = torch.where(torch.argmax(soften_result[0][positive_inds], dim=-1) == labels_pos)[0]
    labels_roi = labels_pos[correct_preds_inds]
    features_roi = roi_align_features[positive_inds[correct_preds_inds]]
    masks_roi = mask_targets[correct_preds_inds]
    
    return labels_roi, features_roi, masks_roi
            
def do_extract(
    args,
    model,
    data_loader,
    device,
    arguments,  # extra parameters, e.g. arguments[iteration]
    n_old_classes=15
    ):
    start_iter = 0
    model.eval() 
    

    old_prototypes = torch.zeros((n_old_classes, n_old_classes))
    n_old_proto = torch.zeros((n_old_classes))
    old_prototypes_classif = torch.zeros((n_old_classes, n_old_classes))
    n_old_proto_classif = torch.zeros((n_old_classes))

    all_labels, all_masked_features, all_binary_masks = [], [], [] # for rehearsal
    
    kept_examplar = None
    kept_examplars = {}
    kept_rois = {}
    for iteration, (images, targets, _, idx) in tqdm(enumerate(data_loader, start_iter), total=len(data_loader)):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]

        with torch.no_grad():
            soften_result, soften_mask_logits, soften_proposal, feature_source, _, _, rpn_output, roi_align_features = model.generate_soften_proposal(images)
        
        """labels_roi, features_roi, masks_roi = get_memory_rois(model, soften_proposal, targets, soften_result, roi_align_features)
        
        import pdb
        pdb.set_trace()

        features_roi.shape
        x_box = model.roi_heads.box.feature_extractor.head(features_roi)
        class_logits, _ = model.roi_heads.box.predictor(x_box)
        torch.argmax(class_logits, dim=-1)

        x_mask = x_box.clone()
        mask_logits = model.roi_heads.mask.predictor(x_mask)
        (mask_logits[0,12]>0.5).int()
        
        print(torch.argmax(mask_logits, dim=-1)[0])
        mask_logits.shape

        classification_loss = F.cross_entropy(class_logits, rehearsal_labels)

        if len(labels_roi) > 0:
            i = np.random.choice(np.arange(len(labels_roi)))
            #for i in range(len(labels_roi)):
            cls = int(labels_roi[i])
            if cls in kept_rois:
                kept_rois[cls].append((features_roi[i].cpu(), masks_roi[i].cpu()))
            else:
                kept_rois[cls] = [(features_roi[i].cpu(), masks_roi[i].cpu())]

        
        batch_features = crop_features(targets[0], feature_source[0])
        
        kept_examplar = filter_examplars(images, batch_features, targets[0], model)
        if kept_examplar is not None:
            cls = int(kept_examplar[0].cpu())
            if cls in kept_examplars:
                if len(kept_examplars[cls]) < 20:
                    kept_examplars[cls].append((kept_examplar[1], kept_examplar[2]))
            else:
                kept_examplars[cls] = [(kept_examplar[1], kept_examplar[2])]"""

        new_protos, labels_idx, labels_counts = extract_segmentation_prototypes(model, soften_proposal, soften_mask_logits, targets)
        old_prototypes, n_old_proto = store_segmentation_prototypes(new_protos, old_prototypes, n_old_proto, labels_idx, labels_counts)
        new_protos_classif, labels_idx_classif, labels_counts_classif = extract_classif_prototypes(model, soften_proposal, soften_result, targets)
        old_prototypes_classif, n_old_proto_classif = store_segmentation_prototypes(new_protos_classif, old_prototypes_classif, n_old_proto_classif, labels_idx_classif, labels_counts_classif)
                
    #with open("old_kept_examplars_instances_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp:
    #    pickle.dump(kept_examplars, fp)


    old_prototypes_norm = old_prototypes / np.linalg.norm(old_prototypes,axis=-1)[:, np.newaxis]
    old_prototypes_classif_norm = old_prototypes_classif / np.linalg.norm(old_prototypes_classif,axis=-1)[:, np.newaxis]
    n_base_classes = int(args.task.split("-")[0])
    n_new_classes_per_step = int(args.task.split("-")[1])
    n_classes_task = n_base_classes+n_new_classes_per_step*(int(args.step)-1)
    #all_masked_features = torch.cat([torch.stack(x) for x in all_masked_features], dim=0)
    #all_labels = np.concatenate([np.stack(x) for x in all_labels], axis=0)

    #sampled_features, sampled_binary_masks = sample_features(n_per_class=20, features=all_masked_features.cpu().numpy(), labels=all_labels, binary_masks=all_binary_masks)
    
    full_name = f"{args.name}"
    output_dir = "mask_out/"+ args.task + "/" + full_name


    #import pickle
    #import pdb
    #pdb.set_trace()
    ## ADD new rois to old ones (instead of reinitializing them at each step)
    #if int(args.step) > 1:
    #    with open("old_kept_rois_{}-{}".format(n_base_classes, n_new_classes_per_step*(int(args.step)-1)), "rb") as fp: old_kept_rois = pickle.load(fp)
    #    for c in range(n_base_classes + n_new_classes_per_step*(int(args.step)-2), n_classes_task):
    #        old_kept_rois[c] = kept_rois[c]
    #    with open(output_dir+"/old_kept_rois_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp: pickle.dump(old_kept_rois, fp)
    #with open(output_dir+"/old_kept_rois_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp: pickle.dump(kept_rois, fp)

    #np.save("old_examplars_{}-{}".format(n_classes_task, n_new_classes_per_step), sampled_features)
    
    #with open("old_examplars_masks_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp: 
    #    pickle.dump(sampled_binary_masks, fp)
    np.save(output_dir+"/old_prototypes_{}-{}".format(n_classes_task, n_new_classes_per_step), old_prototypes_norm) # 15-1, 16-1, 17-1, 18-1, 19-1    
    np.save(output_dir+"/old_prototypes_classif_{}-{}".format(n_classes_task, n_new_classes_per_step), old_prototypes_classif_norm)

def compute_prototypes(cfg, args, distributed):
    
    model = build_detection_model(cfg)

    # default is "cuda"
    device = torch.device(cfg.MODEL.DEVICE)
    # move the model to device
    model.to(device)

    optimizer = make_optimizer(cfg, model)

    arguments = {}
    arguments["iteration"] = 0

    # according to configuration within yaml file sets the learning rate updating strategy
    scheduler = make_lr_scheduler(cfg, optimizer)
    output_dir = ""
    
    checkpointer = DetectronCheckpointer(cfg, model, optimizer, scheduler, output_dir, False)
    print(cfg.MODEL.WEIGHT)
    # load the pre-trained model parameter to current model
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    # dict updating method to update the parameter dictionary
    arguments.update(extra_checkpoint_data)

    # load training data
    # type of data_loader is list, type of its inside elements is torch.utils.data.DataLoader
    # When is_train=True, make sure cfg.DATASETS.TRAIN is a list
    # it has to point to one or multiple annotation files

    data_loader = get_data_loader(cfg)

    # train the model: call function ./maskrcnn_benchmark/engine/trainer.py do_train() function
    do_extract(
        args,
        model,
        data_loader,
        device,
        arguments,
        n_old_classes=len(data_loader.dataset.new_classes)
    )

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "-n", "--name",
        default="EXP",
    )
    parser.add_argument(
        "-c", "--config_file",
        default="../configs/e2e_faster_rcnn_R_50_C4_1x.yaml",
        metavar="FILE", 
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-t", "--task",
        type=str,
        default="15-5"
    )
    parser.add_argument(
        "-s", "--step",
        default=1, type=int
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    
    # if there is more than 1 gpu, set initialization for distribute training
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()
    num_gpus = get_world_size()
    print("I'm using ", num_gpus, " gpus!")

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))

    # open and read the input yaml file, store it on config_str and display on the screen
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cfg.CLS_PER_STEP = int(args.task.split("-")[1])
    if args.step >= 2:
        base = "mask_out"
        cfg.MODEL.SOURCE_WEIGHT = f"{base}/{args.task}/{args.name}/STEP{args.step - 1}/model_trimmed.pth"
        cfg.MODEL.WEIGHT = cfg.MODEL.SOURCE_WEIGHT
    if args.step > 0 and cfg.CLS_PER_STEP != -1:
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES += (args.step - 1) * cfg.CLS_PER_STEP
    else:
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1

    if args.step >= 2:
        base = "mask_out"
        cfg.MODEL.WEIGHT = f"{base}/{args.task}/{args.name}/STEP{args.step - 1}/model_trimmed.pth"
    else:
        base = "mask_out"
        cfg.MODEL.WEIGHT = f"{base}/{args.task}/LR005_BS4_36K/model_trimmed.pth"

    if args.step > 0 and cfg.CLS_PER_STEP != -1:
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES) + 1
        cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES += (args.step - 1) * cfg.CLS_PER_STEP

        cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES += cfg.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES[:(args.step - 1) * cfg.CLS_PER_STEP]
        cfg.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES = cfg.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES[(args.step - 1)* cfg.CLS_PER_STEP:]
        cfg.MIX_UCE = True
        cfg.SOLVER.IMS_PER_BATCH = 1
        print(cfg.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES)
        print(cfg.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)


    
    compute_prototypes(cfg, args, True)

if __name__ == "__main__":
    main()