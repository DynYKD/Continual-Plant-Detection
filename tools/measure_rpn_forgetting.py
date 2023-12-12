from utils import *
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
import os
from maskrcnn_benchmark.data import make_data_loader 
from tqdm import tqdm
import matplotlib.pyplot as plt

cfg, args = get_cfg_args()
model = build_detection_model(cfg)

model.to(cfg.MODEL.DEVICE)
model.eval()

output_dir = cfg.OUTPUT_DIR
checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
_ = checkpointer.load(cfg.MODEL.WEIGHT, resume=False)

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
        output_folders[idx] = output_folder
data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)[0]

def compute_recall(targets, boxes_pred, classwise_recall):
    for i in range(len(targets[0].get_field("labels"))):
        label = targets[0].get_field("labels")[i]
        #boxes_pred = outputs[0][0][0].bbox.cpu()#output[0].bbox.cpu()
        boxes_gt = [targets[0].bbox[i].cpu()]
        IOUs = pairwise_iou(boxes_gt, boxes_pred)
        p, tp, fp, fn = compute_metrics_IOUS(IOUs, 0.5)
        classwise_recall[int(label)-1, 0] += tp
        classwise_recall[int(label)-1, 1] += p
    return classwise_recall

def compute_mask_recall(targets, masks_pred, classwise_recall):
    for i in range(len(targets[0].get_field("labels"))):
        label = targets[0].get_field("labels")[i]
        #boxes_pred = outputs[0][0][0].bbox.cpu()#output[0].bbox.cpu()
        masks_gt = [np.array(targets[0].get_field("masks").instances.masks.cpu()[i])]
        IOUs = pairwise_iou_masks(masks_gt, masks_pred)
        p, tp, fp, fn = compute_metrics_IOUS(IOUs, thresh=0.5)
        classwise_recall[int(label)-1, 0] += tp
        classwise_recall[int(label)-1, 1] += p
    return classwise_recall

def compute_rpn_recall(data_loader):
    rpn_classwise_recall = np.zeros((20,2))
    box_classwise_recall = np.zeros((20,2))
    mask_classwise_recall = np.zeros((20,2))
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, proposals, img_id = batch
        ious = []
        # load images and proposals to gpu
        images = images.to(cfg.MODEL.DEVICE)
        
        with torch.no_grad(): output, features, results_background = model(images)
        with torch.no_grad(): rpn_outputs = model.rpn(images, features)
        #with torch.no_grad(): roi_box_outputs = model.roi_heads(images, features)
        #boxes_pred = 
        rpn_classwise_recall = compute_recall(targets, rpn_outputs[0][0][0].bbox.cpu(), rpn_classwise_recall)
        box_classwise_recall = compute_recall(targets, output[0].bbox.cpu(), box_classwise_recall)
        mask_classwise_recall = compute_mask_recall(targets, np.array(output[0].get_field("mask").cpu()), mask_classwise_recall)
        
        
        if idx % 100 == 0:
            print(rpn_classwise_recall)
            print(box_classwise_recall)
            print(mask_classwise_recall)
        #if idx == 100:
        #    break
    return rpn_classwise_recall, box_classwise_recall
    

rpn_classwise_recall, box_classwise_recall = compute_rpn_recall(data_loader)

import pdb
pdb.set_trace()

all_recall = []
for i in range(20): all_recall.append(rpn_classwise_recall[i,0]/rpn_classwise_recall[i,1])
for i in range(20): print(box_classwise_recall[i,0]/box_classwise_recall[i,1])

for i in range(20): print(all_recall[i], all_recall_baseline[i], all_recall[i] - all_recall_baseline[i])
np.mean(all_recall)
np.mean(all_recall_baseline)
baseline_classwise_recall = np.load("results_forgetting/classwise_recall_mask_out_19-1_MMA_plus_baseline_lr001_STEP1_model_trimmed.pth.npy")

np.save("results_forgetting/box_classwise_recall_{}".format(cfg.MODEL.WEIGHT.replace("/","_")), box_classwise_recall)


import matplotlib.pyplot as plt

plt.bar(x = np.arange(20), height = all_recall)
plt.savefig("visss")
plt.close()
"""
image = cv2.UMat(np.array(images.tensors[0].cpu()).transpose(1,2,0).astype("float32"))
img = overlay_mask(image, output[0])

plt.imshow(img.astype("uint8")[:,:,::-1])
np_img = np.array(img)#.transpose(1,2,0)
np_img[:,:,0] += cfg.INPUT.PIXEL_MEAN[0]
np_img[:,:,1] += cfg.INPUT.PIXEL_MEAN[1]
np_img[:,:,2] += cfg.INPUT.PIXEL_MEAN[2]
plt.imshow(np_img[:,:,::-1].astype("uint8"))
plt.savefig("visss")
plt.close()

masks = output[0].get_field("mask")
plt.imshow(masks[2][0])
plt.savefig("visss")
plt.close()

output[0].get_field("mask").shape"""




    