# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

# This is a backup file in date of 2022-10-20. Weighted distillation was hard-coded for 15-5

import sys
sys.path.append("/lustre06/project/6001581/mapaf2/")
from maskrcnn_benchmark.distillation.attentive_distillation import calculate_attentive_distillation_loss
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import datetime
import logging
import time
import torch
import torch.distributed as dist
from torch import nn
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

from maskrcnn_benchmark.config import \
    cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from maskrcnn_benchmark.data import make_data_loader  # import data set
from maskrcnn_benchmark.engine.inference import inference  # inference
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict  # when multiple gpus are used, reduce the loss
from maskrcnn_benchmark.modeling.detector import build_detection_model  # used to create model
from maskrcnn_benchmark.solver import make_lr_scheduler  # learning rate updating strategy
from maskrcnn_benchmark.solver import make_optimizer  # setting the optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank  # related to multi-gpu training; when usong 1 gpu, get_rank() will return 0
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logger import setup_logger  # related to logging model(output training status)
from maskrcnn_benchmark.utils.miscellaneous import mkdir  # related to folder creation
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from torch.utils.tensorboard import SummaryWriter
from maskrcnn_benchmark.distillation.distillation import calculate_rpn_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_feature_distillation_loss
from maskrcnn_benchmark.distillation.distillation import calculate_roi_distillation_losses
from maskrcnn_benchmark.distillation.distillation import calculate_roi_align_distillation, calculate_mask_distillation_losses
import random
from maskrcnn_benchmark.modeling.utils import cat

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def pad_masks(m):
    if len(m.shape) == 2:
        m = m.unsqueeze(0)
    return m

def pairwise_cosine_sim(x,y):
    assert not torch.isnan(x).any() and not torch.isnan(y).any(), f"Number of nan in x and y: {len(x[torch.isnan(x)])} and {len(y[torch.isnan(y)])}"
    # We fist normalize the rows, before computing their dot products via transposition:
    # x_norm = x.norm(dim=1)
    # x_norm = x_norm[:, None]
    x_norm = x / x.norm(dim=1)[:, None]
    #x_norm = x / x_norm
    y_norm = y / y.norm(dim=1)[:, None]
    y_norm = y_norm.transpose(0,1)

    assert x_norm.device == y_norm.device, f"Device of x {x_norm.device} <> Device of y {y_norm.device}"
    res = torch.mm(x_norm, y_norm)
    res[torch.isnan(res)] = 0
    return res

def do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_source, checkpointer_target,
             device, checkpoint_period, arguments_source, arguments_target, summary_writer, cfg, distributed=False):
    # record log information
    logger = logging.getLogger("maskrcnn_benchmark_target_model.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")  # used to record
    max_iter = len(data_loader)  # data loader rewrites the len() function and allows it to return the number of batches (cfg.SOLVER.MAX_ITER)
    start_iter = arguments_target["iteration"]  #
    print(start_iter)
    model_target.train()  # set the target model in training mode
    model_source.eval()  # set the source model in inference mode
    start_training_time = time.time()
    end = time.time()
    average_distillation_loss = 0
    average_faster_rcnn_loss = 0
    average_weighted_distillation_loss = 0
    weighted_kd_loss = torch.tensor(0) 
    P_0 = np.load("old_prototypes_15-5.npy") #np.random.randn(V, C) # 15 old classes, 21 total
    P_0 = P_0 / np.linalg.norm(P_0, axis=-1)[:,np.newaxis]   
    #P_0 = np.random.randn(15,15)   
    P_0 = torch.Tensor(P_0)
    P_0 = torch.nn.functional.normalize(P_0, dim=-1).detach()

    

    for iteration, (images, targets, _, idx) in tqdm(enumerate(data_loader, start_iter), total=len(data_loader)):
        
        data_time = time.time() - end
        iteration = iteration + 1
        arguments_target["iteration"] = iteration

        images = images.to(device)  # move images to the device
        targets = [target.to(device) for target in targets]  # move targets (labels) to the device

        dist_type = cfg.DIST.TYPE

        with torch.no_grad():
            soften_result, soften_mask_logits, soften_proposal, feature_source, _, _, rpn_output_source, roi_align_features_source = \
                model_source.generate_soften_proposal(images)
        loss_dict_target, feature_target, _, _, rpn_output_target, target_proposals, _, target_soften_results \
            = model_target(images, targets, rpn_output_source=rpn_output_source)
        
        faster_rcnn_losses = sum(loss for loss in loss_dict_target.values())  # summarise the losses for faster rcnn

        # if cfg.DIST.INV_CLS:
        #     soften_result, _ = model_source.forward(images, targets, features=feature_source,
        #                                             proposals=target_proposals)
        #     target_result = target_soften_results
        # else:
        target_result, target_mask_logits, roi_align_features_target = model_target.forward(images, targets,
                                                                                            features=feature_target,
                                                                                            proposals=soften_proposal) 

        import pdb            
        
        if cfg.DIST.WEIGHTED_DIST:
            class ClassSimilarityWeightedKD():
                def __init__(self, prototypes, temperature=3):
                    self.prototypes = prototypes
                    self.temperature = temperature

                def __call__(self, pred_S, pred_T, prototype_map):
                    prototype_map = prototype_map.reshape(-1,15)
                    non_bg_ids = torch.sum(prototype_map, dim=-1)!=0
                    prototype_map = prototype_map[non_bg_ids]
                    r_map = pairwise_cosine_sim(prototype_map.reshape(-1, 15).to("cuda"), self.prototypes.to("cuda"))
                    r_map = torch.nn.functional.softmax(r_map, dim=-1)
                    r_map[r_map < 0] = 0.0

                    pred_S = pred_S[non_bg_ids]
                    pred_T = pred_T[non_bg_ids]
                    #print(pred_S[0])
                    #print(pred_T[0])
                    #print(torch.mean(torch.abs(pred_S - pred_T)))
                    pred_S = torch.nn.functional.log_softmax(pred_S / self.temperature, dim=1)
                    pred_T = torch.nn.functional.softmax(pred_T / self.temperature, dim=1)
                    pred_T = pred_T *  r_map + 10 ** (-7)
                    pred_T = torch.autograd.Variable(pred_T.data.cuda(), requires_grad=False)
                    loss = self.temperature * self.temperature * torch.sum(-pred_T*pred_S)/pred_T.shape[0]
                    if torch.isnan(loss):
                        loss = torch.tensor(0).cuda()
                    #print(loss)
                    return loss

            class BatchPrototypesExtractor:
                def __init__(self, labels_pos, mask_targets, positive_inds):
                    self.labels_pos, self.mask_targets, self.positive_inds = labels_pos, mask_targets, positive_inds

                def __call__(self, target_mask_logits):
                    target_mask_logits = target_mask_logits.permute(0,2,3,1)
                    new_protos = self.extract_proto_features(target_mask_logits)[:,1:16]
                    #new_protos = self.build_new_protos(new_proto_features)[:,1:16]
                    prototype_map = torch.zeros((len(self.positive_inds), 14, 14, 15)).cuda()
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
                    
                    labels_idx = torch.unique(self.labels_pos)
                    new_proto_features = torch.zeros((len(labels_idx), 21)).cuda()
                    for i in range(len(self.positive_inds)):
                        labels.append(self.labels_pos[i])
                        new_proto_features[torch.where(labels_idx==self.labels_pos[i])] += torch.sum(target_mask_logits[self.positive_inds[i]][self.mask_targets[i].bool()], dim=0)
                        #.append(target_mask_logits[self.positive_inds[i]][self.mask_targets[i].bool()])

                    #new_proto_features
                    #new_proto_features = new_proto_features[new_proto_features[:, 0].argsort()]
                    #new_proto_features = np.split(new_proto_features[:,1], np.unique(new_proto_features[:, 0], return_index=True)[1][1:])
                    return new_proto_features

            def get_pos_preds(model_target, soften_proposal, targets):
                labels, mask_targets = model_target.module.roi_heads.mask.loss_evaluator.prepare_targets(soften_proposal, targets)
                labels = cat(labels, dim=0)
                mask_targets = cat(mask_targets, dim=0)
                positive_inds = torch.nonzero(labels > 0).squeeze(1)
                labels_pos = labels[positive_inds]
                return labels_pos, mask_targets, positive_inds

            labels_pos, mask_targets, positive_inds = get_pos_preds(model_target, soften_proposal, targets)

            batch_prototype_extractor = BatchPrototypesExtractor(labels_pos, mask_targets, positive_inds)
            prototype_map = batch_prototype_extractor(target_mask_logits)
            
            weighted_KD = ClassSimilarityWeightedKD(P_0)
            
            pred_S = target_mask_logits[positive_inds].permute(0,2,3,1).reshape(-1, 21)[:, 1:16]
            pred_T = soften_mask_logits[positive_inds].permute(0,2,3,1).reshape(-1, 16)[:, 1:]
            
            weighted_kd_loss = weighted_KD(pred_S, pred_T, prototype_map)

        if cfg.DIST.CLS > 0:
            distillation_losses = cfg.DIST.CLS * calculate_roi_distillation_losses(soften_result, target_result, dist=dist_type)
        else:
            distillation_losses = torch.tensor(0.).to(device)

        if cfg.MODEL.MASK_ON and cfg.DIST.MASK > 0:
            distillation_losses += cfg.DIST.MASK * calculate_mask_distillation_losses(soften_mask_logits, target_mask_logits)

        if cfg.DIST.WEIGHTED_DIST:
            distillation_losses += cfg.DIST.WEIGHTED_DIST * weighted_kd_loss

        if cfg.DIST.RPN:
            rpn_distillation_losses = calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target,
                                                                    cls_loss='filtered_l2', bbox_loss='l2',
                                                                    bbox_threshold=0.1)
            distillation_losses += rpn_distillation_losses
        if cfg.DIST.FEAT == 'align':
            feature_distillation_losses = calculate_roi_align_distillation(roi_align_features_source,
                                                                        roi_align_features_target)
            distillation_losses += feature_distillation_losses
        elif cfg.DIST.FEAT == 'std':
            feature_distillation_losses = calculate_feature_distillation_loss(feature_source, feature_target,
                                                                            loss='normalized_filtered_l1')
            distillation_losses += feature_distillation_losses
        elif cfg.DIST.FEAT == 'att':
            feature_distillation_losses = calculate_attentive_distillation_loss(feature_source[0], feature_target[0])
            distillation_losses += 0.1 * feature_distillation_losses

        distillation_dict = {}
        distillation_dict['distillation_loss'] = distillation_losses.clone().detach()
        weighted_distillation_dict = {}
        weighted_distillation_dict['weighted_distillation_loss'] = weighted_kd_loss.clone().detach()
        loss_dict_target.update(distillation_dict)
        loss_dict_target.update(weighted_distillation_dict)

        losses = faster_rcnn_losses + distillation_losses #distillation_losses# faster_rcnn_losses + distillation_losses
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict_target)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        if (iteration - 1) > 0:
            average_distillation_loss = (average_distillation_loss * (iteration - 1) + distillation_losses) / iteration
            average_faster_rcnn_loss = (average_faster_rcnn_loss * (iteration - 1) + faster_rcnn_losses) / iteration
            average_weighted_distillation_loss = (average_weighted_distillation_loss * (iteration - 1) + weighted_kd_loss) / iteration
        else:
            average_distillation_loss = distillation_losses
            average_faster_rcnn_loss = faster_rcnn_losses
            average_weighted_distillation_loss = weighted_kd_loss

        optimizer.zero_grad()  # clear the gradient cache
        
        # If mixed precision is not used, this ends up doing nothing, otherwise apply loss scaling for mixed-precision recipe.
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()  # use back-propagation to update the gradient
        
        optimizer.step()  # update learning rate
        scheduler.step()  # update the learning rate

        #for i, l in enumerate(model_target.state_dict()):
        #    if torch.sum(model_source.state_dict()[l.replace("module.","")]-model_target.state_dict()[l]):
        #        print(l)
        #    if i == 100:
        #        break

        #print(model_source.state_dict()["backbone.body.layer2.0.conv1.weight"]- model_target.state_dict()["module.backbone.body.layer2.0.conv1.weight"])

        #pdb.set_trace()
        # time used to do one batch processing
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # according to time'moving average to calculate how much time needed to finish the training
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        # for every 50 iterations, display the training status
        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(["eta: {eta}", "iter: {iter}", "{meters}", "lr: {lr:.6f}", "max mem: {memory:.0f}"
                                    ]).format(eta=eta_string, iter=iteration, meters=str(meters),
                                                lr=optimizer.param_groups[0]["lr"],
                                                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))
            # write to tensorboardX
            loss_global_avg = meters.loss.global_avg
            loss_median = meters.loss.median
            # print('loss global average: {0}, loss median: {1}'.format(meters.loss.global_avg, meters.loss.median))
            summary_writer.add_scalar('train_loss_global_avg', loss_global_avg, iteration)
            summary_writer.add_scalar('train_loss_median', loss_median, iteration)
            summary_writer.add_scalar('train_loss_raw', losses_reduced, iteration)
            summary_writer.add_scalar('distillation_losses_raw', distillation_losses, iteration)
            summary_writer.add_scalar('faster_rcnn_losses_raw', faster_rcnn_losses, iteration)
            summary_writer.add_scalar('weighted_distillation_losses_avg', average_weighted_distillation_loss, iteration)
            summary_writer.add_scalar('distillation_losses_avg', average_distillation_loss, iteration)
            summary_writer.add_scalar('faster_rcnn_losses_avg', average_faster_rcnn_loss, iteration)

        # Every time meets the checkpoint_period, save the target model (parameters)
        if iteration % checkpoint_period == 0:
            checkpointer_target.save("model_last", **arguments_target)
        # When meets the last iteration, save the target model (parameters)
        if iteration == max_iter:
            checkpointer_target.save("model_final", **arguments_target)
    # Display the total used training time
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))


def initalizeTargetCls_MiB(cfg, model_source, model_target):
    n_old_classes = len(cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
    cls_score_source = model_source.roi_heads.box.predictor.cls_score
    with torch.no_grad():
        model_target.roi_heads.box.predictor.cls_score.weight[n_old_classes + 1:] = cls_score_source.weight[0]
        model_target.roi_heads.box.predictor.cls_score.bias[n_old_classes + 1:] = \
            cls_score_source.bias[0] - torch.log(torch.Tensor([n_old_classes]).to(cls_score_source.bias.device))
    return model_target


def train(cfg_source, cfg_target, logger_target, distributed, num_gpus, local_rank, remove_old_classes=True):
    print(cfg_source)
    model_source = build_detection_model(cfg_source)  # create the source model
    model_target = build_detection_model(cfg_target)  # create the target model
    device = torch.device(cfg_source.MODEL.DEVICE)  # default is "cuda"
    model_target.to(device)  # move target model to gpu
    model_source.to(device)  # move source model to gpu
    optimizer = make_optimizer(cfg_target, model_target)  # config optimization strategy
    scheduler = make_lr_scheduler(cfg_target, optimizer)  # config learning rate
    # initialize mixed-precision training
    use_mixed_precision = cfg_target.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model_target, optimizer = amp.initialize(model_target, optimizer, opt_level=amp_opt_level)
    # create a parameter dictionary and initialize the iteration number to 0
    arguments_target = {}
    arguments_target["iteration"] = 0
    arguments_source = {}
    arguments_source["iteration"] = 0
    # path to store the trained parameter value
    output_dir_target = cfg_target.OUTPUT_DIR
    output_dir_source = cfg_source.OUTPUT_DIR
    # create summary writer for tensorboard
    summary_writer = SummaryWriter(log_dir=cfg_target.TENSORBOARD_DIR)
    # when only use 1 gpu, get_rank() returns 0
    save_to_disk = get_rank() == 0
    # create check pointer for source model & load the pre-trained model parameter to source model
    checkpointer_source = DetectronCheckpointer(cfg_source, model_source, optimizer=None, scheduler=None,
                                                save_dir=output_dir_source,
                                                save_to_disk=save_to_disk)
    extra_checkpoint_data_source = checkpointer_source.load(cfg_source.MODEL.WEIGHT)
    # create check pointer for target model & load the pre-trained model parameter to target model
    checkpointer_target = DetectronCheckpointer(cfg_target, model_target, optimizer=optimizer, scheduler=scheduler,
                                                save_dir=output_dir_target,
                                                save_to_disk=save_to_disk, logger=logger_target)
    extra_checkpoint_data_target = checkpointer_target.load(cfg_target.MODEL.WEIGHT)
    # dict updating method to update the parameter dictionary for source model
    arguments_source.update(extra_checkpoint_data_source)
    # dict updating method to update the parameter dictionary for target model
    arguments_target.update(extra_checkpoint_data_target)

    # Parameter initialization
    if cfg_target.DIST.INIT:
        model_target = initalizeTargetCls_MiB(cfg_target, model_source, model_target)

    print('start iteration: {0}'.format(arguments_target["iteration"]))

    if distributed:
        model_target = torch.nn.parallel.DistributedDataParallel(
            model_target, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            # broadcast_buffers=False,
        )
    # load training data
    data_loader = make_data_loader(cfg_target, is_train=True, is_distributed=distributed,
                                   start_iter=arguments_target["iteration"], num_gpus=num_gpus, rank=get_rank(),
                                   remove_old_classes=remove_old_classes)
    print('finish loading data')
    # number of iteration to store parameter value in pth file
    checkpoint_period = cfg_target.SOLVER.CHECKPOINT_PERIOD

    # train the model
    
    do_train(model_source, model_target, data_loader, optimizer, scheduler, checkpointer_source, checkpointer_target,
             device, checkpoint_period, arguments_source, arguments_target, summary_writer, cfg_target, distributed)

    checkpointer_target.save("model_trimmed", trim=True, **arguments_target)

    return model_target


def test(cfg):
    if get_rank() != 0:
        return
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    summary_writer = SummaryWriter(log_dir=cfg.TENSORBOARD_DIR)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        result = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            alphabetical_order=cfg.TEST.COCO_ALPHABETICAL_ORDER,
            summary_writer=summary_writer
        )
        if not cfg.MODEL.MASK_ON:
            with open(os.path.join("output", f"{cfg.TASK}.txt"), "a") as fid:
                fid.write(cfg.NAME)
                fid.write(",")
                fid.write(str(cfg.STEP))
                fid.write(",")
                fid.write(",".join([str(x) for x in result["ap"][1:]]))
                fid.write("\n")
        else:
            with open(os.path.join("mask_out", f"{cfg.TASK}_mask.txt"), "a") as fid:
                fid.write(cfg.NAME)
                fid.write(",")
                fid.write(str(cfg.STEP))
                fid.write(",")
                fid.write(",".join([str(x) for x in result['mask']]))
                fid.write("\n")
            with open(os.path.join("mask_out", f"{cfg.TASK}_box.txt"), "a") as fid:
                fid.write(cfg.NAME)
                fid.write(",")
                fid.write(str(cfg.STEP))
                fid.write(",")
                fid.write(",".join([str(x) for x in result['box']]))
                fid.write("\n")

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

    parser.add_argument(
        "-t", "--task",
        type=str,
        default="15-5"
    )
    parser.add_argument(
        "--ist",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "--rpn",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--feat",
        default="no",
        type=str, choices=['no', 'std', 'align', 'att']
    )
    parser.add_argument(
        "--uce",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--init",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--inv",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--mask",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--cls",
        default=1.,
        type=float,
    )
    parser.add_argument(
        "--dist_type",
        default="l2",
        type=str, choices=['uce', 'ce', 'ce_ada', 'ce_all', 'l2', 'none', 'weighted_distill']
    )
    parser.add_argument(
        "-n", "--name",
        default="EXP",
    )
    parser.add_argument(
        "-s", "--step",
        default=0, type=int
    )
    parser.add_argument(
        "--remove-old-classes",
        default=True,
        action='store_false'
    )

    parser.add_argument(
        "--weighted_dist",
        default=0., type=float
    )

    args = parser.parse_args()
    print("step = ", args.step)
    
    if args.ist:
        target_model_config_file = f"configs/IS_cfg/{args.task}/e2e_mask_rcnn_R_50_C4_4x_Target_model.yaml"
    else:
        target_model_config_file = f"configs/OD_cfg/{args.task}/e2e_faster_rcnn_R_50_C4_4x_Target_model.yaml"

    full_name = f"{args.name}/STEP{args.step}"  # if args.step > 1 else args.name

    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    distributed = True
    synchronize()
    num_gpus = get_world_size()
    print("Number of gpus : ", get_world_size())

    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(random.randint(1, 1000))

    cfg_source = cfg.clone()
    cfg_source.merge_from_file(target_model_config_file)
    # if args.step == 2:
    #     cfg_source.MODEL.SOURCE_WEIGHT = f"output/{args.name}/model_final.pth"
    if args.step >= 2:
        base = 'output' if not args.ist else "mask_out"
        cfg_source.MODEL.WEIGHT = f"{base}/{args.task}/{args.name}/STEP{args.step - 1}/model_final.pth"
        cfg_source.MODEL.WEIGHT = cfg_source.MODEL.SOURCE_WEIGHT
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES += (args.step - 1) * cfg_source.CLS_PER_STEP
    else:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
    cfg_source.OUTPUT_DIR += args.task + "/" + full_name + "/SRC"
    cfg_source.TENSORBOARD_DIR += args.task + "/" + full_name
    cfg_source.freeze()

    # LOAD THEN MODIFY PARS FROM CLI
    cfg_target = cfg.clone()
    cfg_target.merge_from_file(target_model_config_file)
    # if args.step == 2:
    #     cfg_target.MODEL.WEIGHT = f"output/{args.name}/model_trimmed.pth"
    if args.step >= 2:
        base = 'output' if not args.ist else "mask_out"
        cfg_target.MODEL.WEIGHT = f"{base}/{args.task}/{args.name}/STEP{args.step - 1}/model_trimmed.pth"
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES += args.step * cfg_target.CLS_PER_STEP
        print(cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES)
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES += cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
                                                          :(args.step - 1) * cfg_target.CLS_PER_STEP]
        print(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES)
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES = cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
                                                              args.step * cfg_source.CLS_PER_STEP:]
        print(cfg_target.MODEL.ROI_BOX_HEAD.NAME_EXCLUDED_CLASSES)
        cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES = cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES[
                                                         (args.step - 1) * cfg_target.CLS_PER_STEP:
                                                         args.step * cfg_source.CLS_PER_STEP]
        print(cfg_target.MODEL.ROI_BOX_HEAD.NAME_NEW_CLASSES)
    
    cfg_target.DIST.MASK = args.mask if args.ist else 0.
    cfg_target.DIST.RPN = args.rpn
    cfg_target.DIST.INV_CLS = args.inv
    cfg_target.DIST.FEAT = args.feat
    if args.cls != -1:
        cfg_target.DIST.CLS = args.cls
    else:
        cfg_target.DIST.CLS = len(cfg_target.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) / \
                                                          cfg_target.MODEL.ROI_BOX_HEAD.NUM_CLASSES
    cfg_target.DIST.TYPE = args.dist_type
    cfg_target.DIST.INIT = args.init
    cfg_target.OUTPUT_DIR += args.task + "/" + full_name
    cfg_target.INCREMENTAL = args.uce
    cfg_target.TENSORBOARD_DIR += args.task + "/" + full_name
    cfg_target.TASK = args.task
    cfg_target.STEP = args.step
    cfg_target.NAME = args.name
    cfg_target.MODEL.WEIGHT = cfg_source.MODEL.WEIGHT
    cfg_target.DIST.WEIGHTED_DIST = args.weighted_dist
    cfg_target.freeze()

    output_dir_target = cfg_target.OUTPUT_DIR
    if output_dir_target:
        mkdir(output_dir_target)
    output_dir_source = cfg_source.OUTPUT_DIR
    if output_dir_source:
        mkdir(output_dir_source)
    tensorboard_dir = cfg_target.TENSORBOARD_DIR
    if tensorboard_dir:
        mkdir(tensorboard_dir)

    if get_rank() == 0:
        logger_target = setup_logger("maskrcnn_benchmark_target_model", output_dir_target, get_rank())
        # logger_target.info("config yaml file for target model: {}".format(target_model_config_file))
        logger_target.info("local rank: {}".format(args.local_rank))
        logger_target.info("Using {} GPUs".format(num_gpus))
        # logger_target.info("Collecting env info (might take some time)")
        # logger_target.info("\n" + collect_env_info())
        # open and read the input yaml file, store it on source config_str and display on the screen
        # with open(target_model_config_file, "r") as cf:
        #     target_config_str = "\n" + cf.read()
        # logger_target.info(target_config_str)
        # logger_target.info("Running with config:\n{}".format(cfg_target))
    else:
        logger_target = None


    # # start to train the modelg
    train(cfg_source, cfg_target, logger_target, distributed, num_gpus, args.local_rank,
            remove_old_classes = args.remove_old_classes)
    # start to test the trained target model
    #"mask_out/15-5/MMA_plus_annotated_bg/STEP0/model_final.pth"
    test(cfg_target)


if __name__ == "__main__":
    main()
