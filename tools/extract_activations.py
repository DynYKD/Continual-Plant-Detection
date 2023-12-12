
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
        compression_not_shuffle=True,
        remove_old_classes=False
    )
    cfg.SOLVER.MAX_ITER = len(data_loader.dataset) // cfg.SOLVER.IMS_PER_BATCH  
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=False,  # whether using multiple gpus to train
        start_iter=0,
        num_gpus=get_world_size(),
        compression_not_shuffle=True,
        rank=get_rank(),
        remove_old_classes=False
    )
    return data_loader

def get_pos_preds(model, soften_proposal, targets):
    labels, mask_targets = model.roi_heads.mask.loss_evaluator.prepare_targets(soften_proposal, targets)
    labels = cat(labels, dim=0)
    mask_targets = cat(mask_targets, dim=0)
    positive_inds = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[positive_inds]
    return labels_pos, mask_targets, positive_inds
            
def do_extract(
    args,
    model,
    model_2,
    model_3,
    model_4,
    data_loader,
    device,
    arguments,  # extra parameters, e.g. arguments[iteration]
    n_old_classes=15
    ):
    start_iter = 0
    model.eval() 
    model_2.eval().cuda()
    model_3.eval().cuda()
    model_4.eval().cuda()
    
    kept_features = {}
    kept_features_2 = {}
    kept_features_3 = {}
    kept_features_4 = {}
    
    for iteration, (images, targets, _, idx) in tqdm(enumerate(data_loader, start_iter), total=len(data_loader)):
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad(): feature_source = model.generate_soften_proposal(images)[3]
        with torch.no_grad(): feature_source_2 = model_2.generate_soften_proposal(images)[3]
        with torch.no_grad(): feature_source_3 = model_3.generate_soften_proposal(images)[3]
        with torch.no_grad(): feature_source_4 = model_4.generate_soften_proposal(images)[3]
           
        downsize_mask = torch.nn.functional.interpolate(targets[0].get_field("masks").instances.masks.unsqueeze(0), feature_source[0].shape[-2:]).squeeze(0)
        
        for i, m in enumerate(downsize_mask): 
            if torch.sum(m) > 0:
                idx = (torch.sum(m)/2).int()
                masked_avg_pool = feature_source[0][:,:, m.bool()].cpu().numpy()#(torch.sum((feature_source[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()#feature_source[0][:,:, m.bool()][:,:,idx].cpu().numpy()#
                masked_avg_pool_2 = feature_source_2[0][:,:, m.bool()].cpu().numpy()#(torch.sum((feature_source_2[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()#feature_source_2[0][:,:, m.bool()][:,:,idx].cpu().numpy()#(torch.sum((feature_source[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()
                masked_avg_pool_3 = feature_source_3[0][:,:, m.bool()].cpu().numpy()#(torch.sum((feature_source_3[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()#feature_source_3[0][:,:, m.bool()][:,:,idx].cpu().numpy()#(torch.sum((feature_source[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()
                masked_avg_pool_4 = feature_source_4[0][:,:, m.bool()].cpu().numpy()#(torch.sum((feature_source_4[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()#feature_source_4[0][:,:, m.bool()][:,:,idx].cpu().numpy()#(torch.sum((feature_source[0] * m), dim=(2,3))/torch.sum(m)).cpu().numpy()
                label = int(targets[0].get_field("labels")[i].cpu())
                if label in kept_features:
                    kept_features[label].append(masked_avg_pool) 
                    kept_features_2[label].append(masked_avg_pool_2) 
                    kept_features_3[label].append(masked_avg_pool_3) 
                    kept_features_4[label].append(masked_avg_pool_4) 
                else:
                    kept_features[label] = [masked_avg_pool]
                    kept_features_2[label] = [masked_avg_pool_2]
                    kept_features_3[label] = [masked_avg_pool_3]
                    kept_features_4[label] = [masked_avg_pool_4]

        #if feature_source[0].shape[2] >=25 and feature_source[0].shape[3] >=25:
        #    kept_features.append(feature_source[0][0,:,25,25].cpu())

        if iteration > 300:
            break
    
    import pdb
    pdb.set_trace()

    n_base_classes = int(args.task.split("-")[0])
    n_new_classes_per_step = int(args.task.split("-")[1])
    n_classes_task = n_base_classes+n_new_classes_per_step*(int(args.step)-1)

    #kept_features = [k.cpu().numpy() for k in kept_features]
    with open("classwise_activations_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp: pickle.dump(kept_features, fp)
    with open("classwise_activations_finetuning_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp: pickle.dump(kept_features_2, fp)
    with open("classwise_activations_baseline_{}-{}".format(n_classes_task, n_new_classes_per_step), "wb") as fp: pickle.dump(kept_features_3, fp)
    with open("classwise_activations_{}-{}".format(n_classes_task-5, n_new_classes_per_step), "wb") as fp: pickle.dump(kept_features_4, fp)
    

def compute_prototypes(cfg, args, distributed):
    from copy import deepcopy
    model = build_detection_model(cfg)
    model_2 = build_detection_model(cfg)
    model_3 = build_detection_model(cfg)
    model_4 = build_detection_model(cfg)

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
    cfg_2 = deepcopy(cfg)
    cfg_3 = deepcopy(cfg)
    cfg_4 = deepcopy(cfg)
    cfg_2.MODEL.WEIGHT = "mask_out/15-1/MMA_plus_weighted_distillation_mask5_wd0_delta0_temp3_simtemp1_cls5_b8_lr001_official_uncenewclassesremoved/STEP5/model_trimmed.pth" #"mask_out/15-1/fine_tuning/STEP1/model_trimmed.pth"#"mask_out/15-5/fine_tuning/STEP1/model_trimmed.pth"
    cfg_3.MODEL.WEIGHT = "mask_out/15-1/MMA_plus_baseline/STEP5/model_trimmed.pth"#"mask_out/15-5/MMA_plus_rerun_baseline/STEP0/model_trimmed.pth"
    cfg_4.MODEL.WEIGHT = "mask_out/15-5/LR005_BS4_36K/model_trimmed.pth"
    checkpointer2 = DetectronCheckpointer(cfg_2, model_2, optimizer, scheduler, output_dir, False)
    checkpointer3 = DetectronCheckpointer(cfg_3, model_3, optimizer, scheduler, output_dir, False)
    checkpointer4 = DetectronCheckpointer(cfg_4, model_4, optimizer, scheduler, output_dir, False)
    print(cfg.MODEL.WEIGHT)
    print(cfg_2.MODEL.WEIGHT)
    # load the pre-trained model parameter to current model
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
    extra_checkpoint_data_2 = checkpointer2.load(cfg_2.MODEL.WEIGHT)
    extra_checkpoint_data_3 = checkpointer3.load(cfg_3.MODEL.WEIGHT)
    extra_checkpoint_data_4 = checkpointer4.load(cfg_4.MODEL.WEIGHT)
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
        model_2,
        model_3,
        model_4,
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
        "--checkpoint",
        type=str,default=None
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
        if args.checkpoint is not None:
            cfg.MODEL.WEIGHT = args.checkpoint
        else:
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