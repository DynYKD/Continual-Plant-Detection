# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F

from maskrcnn_benchmark.layers import smooth_l1_loss
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNLossComputation(object):
    """
    Computes the loss for Faster R-CNN.
    Also supports FPN
    """

    def __init__(
        self, 
        proposal_matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg=False,
        uce=False,
        old_classes=[],
        mix_uce=False
    ):
        """
        Arguments:/
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.uce = uce
        self.mix_uce = mix_uce
        self.n_old_cl = len(old_classes)

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, regression_targets = self.prepare_targets(proposals, targets)
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, regression_targets_per_image, proposals_per_image in zip(
            labels, regression_targets, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field(
                "regression_targets", regression_targets_per_image
            )

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
            zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])
            box_regression (list[Tensor])

        Returns:
            classification_loss (Tensor)
            box_loss (Tensor)
        """
        # print('box_head | loss.py | class_logits size {0}'.format(class_logits[0].size()))
        # print('box_head | loss.py | box_regression size {0}'.format(box_regression[0].size()))
        class_logits = cat(class_logits, dim=0)
        box_regression = cat(box_regression, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        regression_targets = cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )
        
        classification_loss = self.compute_cls_loss(class_logits, labels)

        # get indices that correspond to the regression targets for the corresponding ground truth labels, to be used
        # with advanced indexing
        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        if self.cls_agnostic_bbox_reg:
            map_inds = torch.tensor([4, 5, 6, 7], device=device)
        else:
            map_inds = 4 * labels_pos[:, None] + torch.tensor([0, 1, 2, 3], device=device)

        box_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,  # sum
            beta=1,
        )
        box_loss = box_loss / labels.numel()

        return classification_loss, box_loss

    def compute_cls_loss(self, class_logits, labels):
        if self.mix_uce:
            classification_loss = self.compute_mix_uce(class_logits, labels)
        elif self.uce:
            classification_loss = self.compute_uce_loss(class_logits, labels, reduction="mean")
        else:
            classification_loss = F.cross_entropy(class_logits, labels)
        #print(torch.unique(labels))
        #print(torch.unique(torch.argmax(class_logits, dim=1)))
        return classification_loss

    def compute_uce_loss(self, class_logits, labels, reduction="mean"):
        outputs = torch.zeros_like(class_logits)
        den = torch.logsumexp(class_logits, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(class_logits[:, 0:self.n_old_cl+1], dim=1) - den  # B, H, W       p(O)
        outputs[:, self.n_old_cl+1:] = class_logits[:, self.n_old_cl+1:] - den.unsqueeze(dim=1)  # B, N, H, W    p(N_i)

        labels = labels.clone()  # B, H, W
        classification_loss = F.nll_loss(outputs, labels, reduction=reduction)
        return classification_loss

    def compute_mix_uce(self, class_logits, labels):
        idx_past_classes = torch.isin(labels, torch.range(1, self.n_old_cl).cuda())
        idx_new_classes = torch.logical_not(torch.isin(labels, torch.range(1, self.n_old_cl).cuda()))
        
        new_labels = labels[idx_new_classes]
        past_labels = labels[idx_past_classes]
        
        new_class_logits = class_logits[idx_new_classes]
        past_class_logits = class_logits[idx_past_classes]
        classification_loss = self.compute_uce_loss(new_class_logits, new_labels, reduction='sum')
        if len(past_labels) > 0:   
            classification_loss += F.cross_entropy(past_class_logits, past_labels, reduction='sum')
        if len(labels) > 0 :
            classification_loss /= len(labels)
        
        #if torch.isnan(classification_loss):
        #    classification_loss = torch.tensor(0).cuda()
        return classification_loss

def make_roi_box_loss_evaluator(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

    loss_evaluator = FastRCNNLossComputation(
        matcher, 
        fg_bg_sampler, 
        box_coder, 
        cls_agnostic_bbox_reg,
        cfg.INCREMENTAL,
        cfg.MODEL.ROI_BOX_HEAD.NAME_OLD_CLASSES,
        cfg.MIX_UCE
    )

    return loss_evaluator
