# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import sys
sys.path.append("/lustre06/project/6001581/mapaf2/")

import argparse
import os
import torch
import numpy as np
from maskrcnn_benchmark.config import \
    cfg  # import default model configuration: config/defaults.py, config/paths_catalog.py, yaml file
from maskrcnn_benchmark.data import make_data_loader  # import data set
from maskrcnn_benchmark.engine.inference import inference  # inference
from maskrcnn_benchmark.modeling.detector import build_detection_model  # used to create model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.comm import synchronize, \
    get_rank  # related to multi-gpu training; when usong 1 gpu, get_rank() will return 0
from maskrcnn_benchmark.utils.miscellaneous import mkdir  # related to folder creation
from maskrcnn_benchmark.utils.comm import get_world_size
from torch.utils.tensorboard import SummaryWriter
import random
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def test(cfg):
    if get_rank() != 0:
        return
    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    cfg.MODEL.WEIGHT = "output/26-20/dynykd_b4_lr001_rpn_10K_cls1/STEP1/model_final.pth"
    _ = checkpointer.load(cfg.MODEL.WEIGHT, resume=False)

    def load_backbone(weights_backbone):
        source_weights = torch.load(weights_backbone)["model"]
        source_backbone_weights = {w.replace("module.",""): source_weights[w] for w in source_weights if ("backbone" in w)}
        return source_backbone_weights

    def load_head(weights_1, model=None, w_avg=1):
        weights = torch.load(weights_1)["model"]
        head_weights = {w.replace("module.",""): weights[w] for w in weights if ("backbone" not in w)}
        if model is not None:
            head_weights = {w: w_avg*head_weights[w].cuda() + (1-w_avg)*model.state_dict()[w] for w in head_weights if head_weights[w].shape == model.state_dict()[w].shape} # weights averaging
        return head_weights
    
    def load_multihead(w1, w2, w3, w4, w5):
        weights1 = torch.load(w1)["model"]
        weights2 = torch.load(w2)["model"]
        weights3 = torch.load(w3)["model"]
        weights4 = torch.load(w4)["model"]
        weights5 = torch.load(w5)["model"]
        weights1 = {w.replace("module.",""): weights1[w] for w in weights1 if ("backbone" not in w)}
        weights2 = {w.replace("module.",""): weights2[w] for w in weights2 if ("backbone" not in w)}
        weights3 = {w.replace("module.",""): weights3[w] for w in weights3 if ("backbone" not in w)}
        weights4 = {w.replace("module.",""): weights4[w] for w in weights4 if ("backbone" not in w)}
        weights5 = {w.replace("module.",""): weights5[w] for w in weights5 if ("backbone" not in w)}
        if model is not None:
            head_weights = {w: 0.*weights1[w].cuda() + 0.*weights2[w].cuda()+ 0.33*weights3[w].cuda()+ 0.33*weights4[w].cuda()+ 0.33*weights5[w].cuda() for w in weights5 if weights5[w].shape == weights4[w].shape} # weights averaging
        return head_weights

    weights_backbone = "output/26-20/LR005_BS4_50K/model_final.pth"#"mask_out/10-2/same_backbone_mask1_cls1_b4_lr001/STEP1/model_final.pth"#"mask_out/15-1/MMA_mask2_wd0_temp3_simtemp3_cls2_b4_lr001_same_backbone/STEP1/model_final.pth"#"mask_out/10-5/same_backbone_mask1_cls1.5_b4_lr001/STEP1/model_final.pth"
    #weights_head_4 = "mask_out/15-1/MMA_mask2_wd0_temp3_simtemp3_cls2_b4_lr001_same_backbone/STEP4/model_final.pth"
    model.load_state_dict(load_backbone(weights_backbone), strict=False)
    #model.load_state_dict(load_head(weights_head_4, model, w_avg=0.25), strict=False)

    """n_new_weights = 0
    for k, v in load_backbone(weights_backbone).items(): 
        if abs(torch.sum(v) - torch.sum(model.state_dict()[k])) > 0.1:
            print(k)
            n_new_weights += v.size().numel()


    total_weights = 0
    for k, v in model.state_dict().items():
        for s in v.reshape(-1).size(): 
            total_weights += s"""

    #def count_parameters(model):return sum(p.numel() for p in model.parameters())# if p.requires_grad)
    #def count_parameters(dic):return sum(dic[p].numel() for p in dic.keys())
    #def count_parameters(model):return sum(p.numel() for p in model.values())
    #count_parameters(model)


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
            save_predictions=False,
            summary_writer=summary_writer
        )

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
        default=1, type=int
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
    parser.add_argument(
        "--temperature",
        default=1., type=float
    )
    parser.add_argument(
        "--sim_temperature",
        default=1., type=float
    )
    parser.add_argument(
        "--mix-uce",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--prob-add-roi",
        default=0., type=float
    )
    parser.add_argument(
        "--n-add-roi",
        default=0, type=int
    )
    parser.add_argument(
        "--dataset",
        default="OPPD",
        type=str
    )
    args = parser.parse_args()
    print("step = ", args.step)
    
    if args.ist:
        target_model_config_file = f"configs/IS_cfg/{args.task}/e2e_mask_rcnn_R_50_C4_4x_Target_model.yaml"
    else:
        target_model_config_file = f"configs/OD_cfg/{args.dataset}/{args.task}/e2e_faster_rcnn_R_50_C4_4x_Target_model.yaml"

    full_name = f"{args.name}/STEP{args.step}"  # if args.step > 1 else args.name
    args.local_rank = int(os.environ['LOCAL_RANK'])
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
    
    if args.step >= 2:
        base = 'output' if not args.ist else "mask_out"
        cfg_source.MODEL.SOURCE_WEIGHT = f"{base}/{args.task}/{args.name}/STEP{args.step - 1}/model_trimmed.pth"
        cfg_source.MODEL.WEIGHT = cfg_source.MODEL.SOURCE_WEIGHT
         
    if args.step > 0 and cfg_source.CLS_PER_STEP != -1:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES += (args.step - 1) * cfg_source.CLS_PER_STEP
    else:
        cfg_source.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(cfg_source.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES) + 1
    cfg_source.OUTPUT_DIR += args.task + "/" + full_name + "/SRC"
    cfg_source.TENSORBOARD_DIR += args.task + "/" + full_name
    cfg_source.MIX_UCE = args.mix_uce
    cfg_source.freeze()

    # LOAD THEN MODIFY PARS FROM CLI
    cfg_target = cfg.clone()
    cfg_target.merge_from_file(target_model_config_file)

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
    cfg_target.DIST.TEMPERATURE = args.temperature
    cfg_target.DIST.SIM_TEMPERATURE = args.sim_temperature
    cfg_target.MIX_UCE = args.mix_uce
    cfg_target.USE_ROI_REHEARSAL = bool(args.prob_add_roi > 0)
    cfg_target.PROB_ADD_ROI = args.prob_add_roi
    cfg_target.N_ADD_ROI = args.n_add_roi

    #cfg_target.freeze()
    
    output_dir_target = cfg_target.OUTPUT_DIR
    if output_dir_target:
        mkdir(output_dir_target)
    output_dir_source = cfg_source.OUTPUT_DIR
    if output_dir_source:
        mkdir(output_dir_source)
    tensorboard_dir = cfg_target.TENSORBOARD_DIR
    if tensorboard_dir:
        mkdir(tensorboard_dir)
    else:
        logger_target = None
        
    test(cfg_target)


if __name__ == "__main__":
    main()
