import argparse
import torch
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size
from maskrcnn_benchmark.config import cfg
import numpy as np
import cv2
from maskrcnn_benchmark.utils import cv2_util

def get_parser():
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
    return parser

def get_cfg_args():
    parser = get_parser()
    args = parser.parse_args()

    # if there is more than 1 gpu, set initialization for distribute training
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://"
    )
    synchronize()
    num_gpus = get_world_size()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR

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
    
    return cfg, args


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_mask(image, predictions):
    """
    Adds the instances contours for each predicted object.
    Each label has a different color.
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `mask` and `labels`.
    """
    masks = predictions.get_field("mask").cpu().numpy()
    labels = predictions.get_field("labels").cpu()
    
    colors = compute_colors_for_labels(labels).tolist()

    for mask, color in zip(masks, colors):
        thresh = mask[0, :, :, None].astype(np.uint8)
        contours, hierarchy = cv2_util.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image = cv2.drawContours(image, contours, -1, color, 3)

    composite = cv2.UMat.get(image)

    return composite


def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou
    
def pairwise_iou(bboxes_gt, bboxes_pred):
    results = np.zeros((len(bboxes_gt), len(bboxes_pred)))
    for i, bgt in enumerate(bboxes_gt):
        for j, bpred in enumerate(bboxes_pred):
            results[i, j] = bb_intersection_over_union(bgt, bpred)
    return results

def compute_metrics_IOUS(IOUs, thresh=0.5):
    """Return amount of positives, true positives, false positives and false negatives
        from pairwise IOUs matrix. IOU > thresh is considered a positive prediction.
    """
    IOUs = IOUs + np.random.rand(IOUs.shape[0], IOUs.shape[1]) * 1e-6
    P = IOUs.shape[0]
    predicted_pos = IOUs.shape[1]
    TP = 0
    indices_tp = []
    indices_fn = np.arange(P)
    indices_fp = np.arange(predicted_pos)
    indices_tp_axis0 = []

    if (IOUs.shape[0] > 0) & (IOUs.shape[1] > 0):
        if IOUs.shape[0] < IOUs.shape[1]:
            # S'il y a plus de brocolis que de prédictions, on choisit les prédictions ayant le plsu haut score
            tp_array = np.ones_like(IOUs) * (IOUs.max(axis=0,keepdims=1) == IOUs)
            tp_array = tp_array * IOUs
            max_values = np.max(tp_array, axis=1)
            TP = len(max_values[max_values>thresh])
            indices_tp = np.where(max_values>thresh)[0]
            indices_tp_axis0 = np.where(tp_array * (np.max(tp_array, axis=1,keepdims=1) == IOUs)>thresh)[-1] # np.where(np.max(tp_array, axis=1)>thresh)[0]#np.argmax(tp_array, axis=-1)
            indices_fn = list(set(indices_fn).difference(set(indices_tp)))
            indices_fp = list(set(indices_fp).difference(set(indices_tp_axis0)))
        elif IOUs.shape[0] >= IOUs.shape[1]: 
            for i in range(IOUs.shape[1]):
                m = np.argsort(IOUs[:,i])#np.max(IOUs[:,i])
                for j in reversed(range(len(m))):
                    if IOUs[m[j], i] > thresh:
                        if i not in indices_tp_axis0 and m[j] not in indices_tp:
                            indices_tp_axis0.append(i)
                            indices_tp.append(m[j])
                            break

            #indices_fn = list(set(indices_fn).difference(set(indices_tp_axis0)))
            indices_fn = list(set(indices_fn).difference(set(indices_tp)))
            indices_fp = list(set(indices_fp).difference(set(indices_tp_axis0)))
            #indices_fp = list(set(indices_fp).difference(set(indices_tp)))
            TP = len(indices_tp)
            
        else:
            tp_array = np.ones_like(IOUs) * (IOUs.max(axis=0,keepdims=1) == IOUs)
            tp_array = tp_array * IOUs
            max_values = np.max(tp_array, axis=0)
            indices_tp = np.where(np.max(tp_array, axis=1)>thresh)[0]
            indices_tp_axis0 = np.where(np.max(tp_array, axis=0)>thresh)[0]
            indices_fn = list(set(indices_fn).difference(set(indices_tp)))
            indices_fp = list(set(indices_fp).difference(set(indices_tp_axis0)))
            TP = len(max_values[max_values>thresh])

        assert TP/P <= 1, IOUs
    FP = predicted_pos - TP
    FN = P - TP
    assert (P >= 0) & (TP >= 0) & (FP >= 0) & (FN >= 0), (P, TP, FP, FN)
    return P, TP, FP, FN#, indices_tp_axis0, indices_fn, indices_fp

def masks_intersection_over_union(mask_gt, mask_pred):
    interArea = np.sum(mask_gt * mask_pred)
    iou = interArea / (np.sum(mask_gt) + np.sum(mask_pred) - interArea)
    return iou

def pairwise_iou_masks(masks_gt, masks_pred):
    results = np.zeros((len(masks_gt), len(masks_pred)))
    for i, bgt in enumerate(masks_gt):
        for j, bpred in enumerate(masks_pred):
            results[i, j] = masks_intersection_over_union(bgt, bpred)
    return results