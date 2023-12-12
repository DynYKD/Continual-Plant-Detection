# Pseudo-algo for build-up active learning scheme
import numpy as np
from copy import deepcopy

class Buildup():
    def __init__(self, Ns, f, method="random", device=None, buildup_step=0, seed=42):
        #np.random.seed(seed)
        self.Ns = Ns
        self.f = f
        self.buildup_step = buildup_step
        self.idx_annotated = []
        self.method = method
        self.model = None
        self.device = device
        #self.class_counts = np.zeros((1696))# [[np.zeros((21))] for _ in range(1696)]
        #elf.last_predictions = np.zeros((1696, 21))
        self.epoch_predictions = np.zeros((len(f), 1696, 21))
    def build_training_subset(self, idx, data_loader=None):
        subset_indices = []
        if self.buildup_step < 0:
            subset_indices = self.initialize_dataset(idx)
            self.buildup_step += 1

        elif self.buildup_step >= 0 and len(self.idx_annotated) < self.Ns:
            subset_indices = self.increment_subset(idx, data_loader=data_loader)
            self.buildup_step += 1

        return subset_indices

    def initialize_dataset(self, idx):
        """
        Takes the indices of VOC and randomly sample Ns * f data points.
        args
            idx: list of valid indices (computed in PascalVOCDataset2012)
        """
        indices = deepcopy(idx)
        np.random.shuffle(indices)
        n_subset = int(self.Ns * self.f[0])
        subset_indices = indices[:n_subset]
        self.idx_annotated = list(subset_indices)
        return subset_indices

    def balance_dataset(self, data_loader):
        pass
    
    def increment_subset(self, idx, data_loader=None):
        
        valid_idx = list(set(idx).difference(set(self.idx_annotated)))
        uncertainties = self.compute_uncertainty(idx, data_loader=data_loader, method=self.method)
        sorted_uncrt = np.argsort(uncertainties)[::-1] # decreasing order
        
        max_annotations = self.Ns * self.f[self.buildup_step] # e.g Ns = 1000, f = [1/8, 1/4, 1/2, 1], buildup_step = 1 --> 1000 * 1/4  = 250
        i = 0
        # = np.sort(idx)
        while len(self.idx_annotated) < max_annotations:
            #print(idx[sorted_uncrt[i]], uncertainties[sorted_uncrt[i]])
            if idx[sorted_uncrt[i]] in valid_idx:
                new_anno = idx[sorted_uncrt[i]]
                self.idx_annotated.extend([new_anno])
            i += 1
        assert len(self.idx_annotated) == len(self.idx_annotated)
        for chosen_idx in self.idx_annotated:
            assert chosen_idx in idx
        return self.idx_annotated
        
    def compute_uncertainty(self, idx, data_loader=None, method="random"):
        if self.method == "random":
            print("Using method random")
            uncertainties = self._uncertainty_random(idx)
        elif self.method == "least_confident":
            uncertainties = self._uncertainty_least_confident(data_loader)
        elif self.method == "label_dispersion":
            uncertainties = self._uncertainty_label_dispersion(idx, data_loader)
            #uncertainties = self._uncertainty_random(idx)
        
        return uncertainties

    def _uncertainty_random(self, idx):
        uncertainties = []
        for _ in idx:
            uncertainties.append(np.random.rand())
        return uncertainties

    def _sort_array_from_img_ids(self, array, img_ids):
        sorted_img_ids = np.argsort(img_ids)
        array = np.array(array)[sorted_img_ids]
        return array

    def _uncertainty_least_confident(self, data_loader):
        labels, scores, img_ids = predict(self.model, data_loader, self.device)
        for l, s in zip(labels, scores):
            s = s[l>15]
            if len(s) > 0:
                m = np.min(s.to("cpu").numpy().reshape(-1))
            else:
                m = 1
            uncertainties.append(1-m)
        uncertainties = list(self._sort_array_from_img_ids(uncertainties, img_ids))
        print("uncertainties: ", np.sort(uncertainties))
        return uncertainties

    def _uncertainty_label_dispersion(self, idx, data_loader):
        labels, scores, img_ids = predict(self.model, data_loader, self.device)
        I = np.eye(21)
        for j, l in enumerate(labels):
            indice = img_ids[j]
            diff_count = np.sum([I[l[i].astype("int")] for i in range(len(l))], axis=0)
            self.epoch_predictions[self.buildup_step, indice] = diff_count
        c_counts = self.epoch_predictions
        abs_diff = np.zeros((c_counts.shape[1:]))
        for i in range(1, self.buildup_step+1):
            abs_diff += np.abs(c_counts[i] - c_counts[i-1]).astype("int")
        uncertainties = np.sum(abs_diff, axis=-1)
        print("uncertainties: ", np.sort(uncertainties), np.argsort(uncertainties))
        return uncertainties
        




    def compute_entropy(self, pred):
        uncertainty_map = np.zeros_like(pred)

        for d in range(pred.shape[0]):
            neg_pred = 1 - pred[d]
            entropy = - (neg_pred * np.log2(neg_pred) + pred[d]*np.log2(pred[d]))
            uncertainty_map[d] = entropy

        return np.max(uncertainty_map)



import torch
from tqdm import tqdm
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
def predict(model, data_loader, device, summary_writer=None):
    NAME_CLASSES = np.array(["#bkg","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                       "tvmonitor"])
    model.eval()
    all_labels = []
    all_scores = []
    all_img_ids = []
    for idx, batch in enumerate(tqdm(data_loader)):
        images, targets, proposals, img_id = batch
        # load images and proposals to gpu
        images = images.to(device)
        with torch.no_grad():
            output, features, results_background = model(images)
            if len(output) > 1:
                print(f"Warning: You are testing with BS > 1, but it'll not work!!")
            if len(output[0]) == 0:
                print(f"Warning: Mask of {img_id} has 0 bboxes!")
            masks = output[0].get_field("mask")
            masks = masks.squeeze(dim=1)
            scores = output[0].get_field("scores")
            labels = output[0].get_field("labels")
            if masks.shape[0] > 0 and masks.max() < 0.01:
                print(f"Warning: Masks of {img_id} have max < 0.01!")
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_img_ids.append(img_id[0])

    return all_labels, all_scores, all_img_ids