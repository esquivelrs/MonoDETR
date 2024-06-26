"""
MonoDETR: Depth-aware Transformer for Monocular 3D Object Detection
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
import os
import cv2 as cv

from typing import List
from utils import box_ops, draw3Dbox2img
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list,
                            accuracy, get_world_size, interpolate,
                            is_dist_avail_and_initialized, inverse_sigmoid)

import utils.draw3Dbox2img as viz_utils

from .utils_cubercnn import box_utils
from .utils_cubercnn.box_utils import generate_corners3d, get_heading_angle, alpha2ry, chamfer_loss

from .backbone import build_backbone
from .matcher import build_matcher
from .depthaware_transformer import build_depthaware_transformer, DepthAwareTransformer
from .depth_predictor import DepthPredictor
from .depth_predictor.ddn_loss import DDNLoss
from lib.losses.focal_loss import sigmoid_focal_loss
from .dn_components import prepare_for_dn, dn_post_process, compute_dn_loss

from lib.datasets.kitti.kitti_utils import Calibration, draw_projected_box3d


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MonoDETR(nn.Module):
    """ This is the MonoDETR module that performs monocualr 3D object detection """
    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,
                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, 
                 group_num=11, two_stage_dino=False, depth_predictor2loss = True):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            depthaware_transformer: depth-aware transformer architecture. See depth_aware_transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For KITTI, we recommend 50 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage MonoDETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.depthaware_transformer:DepthAwareTransformer = depthaware_transformer
        self.depth_predictor:DepthPredictor = depth_predictor
        hidden_dim = self.depthaware_transformer.d_model
        self.hidden_dim = hidden_dim
        self.num_feature_levels = num_feature_levels
        self.two_stage_dino = two_stage_dino
        self.label_enc = nn.Embedding(num_classes + 1, hidden_dim - 1)  # # for indicator
        # prediction heads
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.dim_embed_3d = MLP(hidden_dim, hidden_dim, 3, 2)
        self.angle_embed = MLP(hidden_dim, hidden_dim, 24, 2)
        self.depth_embed = MLP(hidden_dim, hidden_dim, 2, 2)  # depth and deviation
        self.use_dab = use_dab
        self.depth_predictor2loss = depth_predictor2loss

        if init_box == True:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if not two_stage:
            if two_stage_dino:
                self.query_embed = None
            if not use_dab:
                self.query_embed = nn.Embedding(num_queries * group_num, hidden_dim*2)
            else:
                self.tgt_embed = nn.Embedding(num_queries * group_num, hidden_dim)
                self.refpoint_embed = nn.Embedding(num_queries * group_num, 6)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels = backbone.num_channels[i]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.num_classes = num_classes

        if self.two_stage_dino:
            _class_embed = nn.Linear(hidden_dim, num_classes)
            _bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
            # init the two embed layers
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            _class_embed.bias.data = torch.ones(num_classes) * bias_value
            nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
            self.depthaware_transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)
            self.depthaware_transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (depthaware_transformer.decoder.num_layers + 1) if two_stage else depthaware_transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.depthaware_transformer.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.depthaware_transformer.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.depthaware_transformer.decoder.bbox_embed = None

        if two_stage:
            # hack implementation for two-stage
            self.depthaware_transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)


    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """


        features, pos = self.backbone(images)

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = torch.zeros(src.shape[0], src.shape[2], src.shape[3]).to(torch.bool).to(src.device)
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.training:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 4
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)

            else:
                tgt_all_embed=tgt_embed = self.tgt_embed.weight[:self.num_queries]
                refanchor = self.refpoint_embed.weight[:self.num_queries]
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
        elif self.two_stage_dino:
            query_embeds = None
        else:
            if self.training:
                query_embeds = self.query_embed.weight
            else:
                # only use one group in inference
                query_embeds = self.query_embed.weight[:self.num_queries]

        ###### DEPTH forwards ######
        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])

        hs, init_reference, inter_references, inter_references_dim, enc_outputs_class, enc_outputs_coord_unact = self.depthaware_transformer(
            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip)#, attn_mask)

        outputs_coords = []
        outputs_classes = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference


            # 3d center + 2d box
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[lvl]
            outputs_3d_dims.append(size3d)

            # depth_geo
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 0, 0].unsqueeze(1)

            # depth_reg
            depth_reg = self.depth_embed[lvl](hs[lvl])

            # depth_map
            outputs_center3d = ((outputs_coord[..., :2] - 0.5) * 2).unsqueeze(2).detach()
            
            if self.depth_predictor2loss:
                depth_map = F.grid_sample(
                    weighted_depth.unsqueeze(1),
                    outputs_center3d,
                    mode='bilinear',
                    align_corners=True).squeeze(1)
                # depth average + sigma
                depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1) + depth_map) / 3,
                                        depth_reg[:, :, 1: 2]], -1)
            else:
                depth_ave = torch.cat([((1. / (depth_reg[:, :, 0: 1].sigmoid() + 1e-6) - 1.) + depth_geo.unsqueeze(-1)) / 3,
                                        depth_reg[:, :, 1: 2]], -1)
            


            outputs_depths.append(depth_ave)

            # angles
            outputs_angle = self.angle_embed[lvl](hs[lvl])
            outputs_angles.append(outputs_angle)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        out['pred_3d_dim'] = outputs_3d_dim[-1]
        out['pred_depth'] = outputs_depth[-1]
        out['pred_angle'] = outputs_angle[-1]
        out['pred_depth_map_logits'] = pred_depth_map_logits

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        return out #, mask_dict

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b,
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for MonoDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, applied_losses, group_num=11):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.applied_losses:List[str] = applied_losses # list with the keys (names) of losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

        self.loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
        }

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True,**kwargs):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx:tuple = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes,**kwargs):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes,**kwargs):
        idx = self._get_src_permutation_idx(indices)
        src_3dcenter = outputs['pred_boxes'][:, :, 0: 2][idx]
        target_3dcenter = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_3dcenter = F.l1_loss(src_3dcenter, target_3dcenter, reduction='none')
        losses = {}
        losses['loss_center'] = loss_3dcenter.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes,**kwargs):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_2dboxes = outputs['pred_boxes'][:, :, 2: 6][idx]
        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # l1
        loss_bbox = F.l1_loss(src_2dboxes, target_2dboxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # giou
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcylrtb_to_xyxy(src_boxes),
            box_ops.box_cxcylrtb_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes,**kwargs):

        idx = self._get_src_permutation_idx(indices)

        src_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = src_depths[:, 0], src_depths[:, 1]
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes
        return losses

    def loss_dims(self, outputs, targets, indices, num_boxes,**kwargs):

        idx = self._get_src_permutation_idx(indices)
        src_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(src_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(src_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes,**kwargs):

        idx = self._get_src_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')

        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes,**kwargs):
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()

        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss_name:str, outputs:dict, targets:dict, indices, num_boxes, **kwargs)->dict:
        """
        returns a dictionary ('loss_<loss_name>', loss_value),
            except for depth_loss which return a 'DDNLoss' object
        """
        # loss_map = {
        #     'labels': self.loss_labels,
        #     'cardinality': self.loss_cardinality,
        #     'boxes': self.loss_boxes,
        #     'depths': self.loss_depths,
        #     'dims': self.loss_dims,
        #     'angles': self.loss_angles,
        #     'center': self.loss_3dcenter,
        #     'depth_map': self.loss_depth_map,
        # }

        assert loss_name in self.loss_map, f'{loss_name} not found in loss map.'

        return self.loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        losses = {}
        for loss_name in self.applied_losses:
            #ipdb.set_trace()
            losses.update(self.get_loss(loss_name, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_num=group_num)
                for loss_name in self.applied_losses:
                    if loss_name == 'depth_map':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss_name == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if loss_name == 'cbr_3d':
                        continue
                    l_dict = self.get_loss(loss_name, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


class ExtendedSetCriterion(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, applied_losses, group_num=11):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__(num_classes, matcher, weight_dict, focal_alpha, applied_losses, group_num=group_num)

        self.loss_map.update(
            {
                'cbr_loss_depth': self.cbr_loss_depth,
                'cbr_loss_xy': self.cbr_loss_xy,
                'cbr_loss_dims': self.cbr_loss_dims,
                'cbr_loss_ry': self.cbr_loss_ry,
                'cbr_loss_all': self.cbr_loss_all
            }
        )

    def cbr_loss_depth(self, outputs:dict, targets:dict, indices, num_boxes,**kwargs)->dict:
        # Constructing Gt 3d corners
        target_3d_center_cxcy = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_3d_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0) # makes a single tensor for all targets in the batch
        target_ry = torch.cat([t['ry'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_img_size = torch.cat([t['img_size_repeat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_calib_P2 = torch.cat([t['calibs'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # _train_root_path = '/home/iony/DTU/s24/adlcv/project/adlcv_project/models/monoDETR/data/KITTIDataset_reduced/training/'
        # target_calibs = viz_utils.get_calibs(_train_root_path, targets_info['img_id'])

        # img to camera coordinates for all target (matrix form)
        target_cucv = target_3d_center_cxcy*target_img_size
        _cus = target_calib_P2[:,0,2]
        _cvs = target_calib_P2[:,1,2]
        _fus = target_calib_P2[:,0,0]
        _fvs = target_calib_P2[:,1,2]
        _txs = target_calib_P2[:,0,3] / (-_fus)
        _tys = target_calib_P2[:,1,3] / (-_fvs)

        _xs3d = (target_cucv[:,0]-_cus).unsqueeze(-1) * target_depth / _fus.unsqueeze(-1) + _txs.unsqueeze(-1)
        _ys3d = (target_cucv[:,1]-_cvs).unsqueeze(-1) * target_depth / _fvs.unsqueeze(-1) + _tys.unsqueeze(-1)
        _xys3d = torch.stack((_xs3d, _ys3d),dim=1).squeeze()
        
        target_corners3d, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, target_depth)

        # predicted 3d dimensions
        pred_matched_idxs = self._get_src_permutation_idx(indices) # prediction idxs matched to targets

        # z loss
        pred_depth = outputs['pred_depth'][:, :, 0][pred_matched_idxs].unsqueeze(dim=1)
        pred_depth_vertices, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, pred_depth)

        loss_depth = F.l1_loss(target_corners3d, pred_depth_vertices, reduction='none').sum()
        loss_depth /= num_boxes

        return {'cbr_loss_depth': loss_depth}

    def cbr_loss_xy(self, outputs:dict, targets:dict, indices, num_boxes,**kwargs)->dict:
        # Constructing Gt 3d corners
        target_3d_center_cxcy = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_3d_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0) # makes a single tensor for all targets in the batch
        target_ry = torch.cat([t['ry'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_img_size = torch.cat([t['img_size_repeat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_calib_P2 = torch.cat([t['calibs'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # _train_root_path = '/home/iony/DTU/s24/adlcv/project/adlcv_project/models/monoDETR/data/KITTIDataset_reduced/training/'
        # target_calibs = viz_utils.get_calibs(_train_root_path, targets_info['img_id'])

        # img to camera coordinates for all target (matrix form)
        target_cucv = target_3d_center_cxcy*target_img_size
        _cus = target_calib_P2[:,0,2]
        _cvs = target_calib_P2[:,1,2]
        _fus = target_calib_P2[:,0,0]
        _fvs = target_calib_P2[:,1,2]
        _txs = target_calib_P2[:,0,3] / (-_fus)
        _tys = target_calib_P2[:,1,3] / (-_fvs)

        _xs3d = (target_cucv[:,0]-_cus).unsqueeze(-1) * target_depth / _fus.unsqueeze(-1) + _txs.unsqueeze(-1)
        _ys3d = (target_cucv[:,1]-_cvs).unsqueeze(-1) * target_depth / _fvs.unsqueeze(-1) + _tys.unsqueeze(-1)
        _xys3d = torch.stack((_xs3d, _ys3d),dim=1).squeeze()
        
        target_corners3d, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, target_depth)

        # predicted 3d dimensions
        pred_matched_idxs = self._get_src_permutation_idx(indices) # prediction idxs matched to targets

        # u,v loss
        pred_out_bbox = outputs['pred_boxes'][pred_matched_idxs] # cxcylrtb

        # adapted from kitti_utils.Calibration
        _fu = target_calib_P2[:, 0, 0]
        _fv = target_calib_P2[:, 1, 1]
        _cu = target_calib_P2[:, 0, 2]
        _cv = target_calib_P2[:, 1, 2]
        _tx = target_calib_P2[:, 0, 3] / _fu
        _ty = target_calib_P2[:, 1, 3] / _fv

        # scale normalized u,v to image size
        _u = pred_out_bbox[:, 0] * target_img_size[:,0]
        _v = pred_out_bbox[:, 1] * target_img_size[:,1]

        # project to 3d (camera coordinates)
        _x3D = ((_u-_cu)*target_depth.squeeze() / _fu ) + _tx
        _y3D = ((_v-_cv)*target_depth.squeeze() / _fv) + _ty

        pred_3d_center_xy = torch.cat((_x3D.reshape(-1, 1), _y3D.reshape(-1, 1)), dim=1)
        pred_xy_vertices, _ = generate_corners3d(target_3d_dims, target_ry, pred_3d_center_xy, target_depth)

        loss_xy = F.l1_loss(target_corners3d, pred_xy_vertices, reduction='none').sum()
        loss_xy /= num_boxes

        return {'cbr_loss_xy': loss_xy}

    def cbr_loss_dims(self, outputs:dict, targets:dict, indices, num_boxes,**kwargs)->dict:
        # Constructing Gt 3d corners
        target_3d_center_cxcy = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_3d_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_ry = torch.cat([t['ry'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_img_size = torch.cat([t['img_size_repeat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_calib_P2 = torch.cat([t['calibs'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # img to camera coordinates for all target (matrix form)
        target_cucv = target_3d_center_cxcy*target_img_size
        _cus = target_calib_P2[:,0,2]
        _cvs = target_calib_P2[:,1,2]
        _fus = target_calib_P2[:,0,0]
        _fvs = target_calib_P2[:,1,2]
        _txs = target_calib_P2[:,0,3] / (-_fus)
        _tys = target_calib_P2[:,1,3] / (-_fvs)

        _xs3d = (target_cucv[:,0]-_cus).unsqueeze(-1) * target_depth / _fus.unsqueeze(-1) + _txs.unsqueeze(-1)
        _ys3d = (target_cucv[:,1]-_cvs).unsqueeze(-1) * target_depth / _fvs.unsqueeze(-1) + _tys.unsqueeze(-1)
        _xys3d = torch.stack((_xs3d, _ys3d),dim=1).squeeze()
        
        target_corners3d, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, target_depth)

        # predicted 3d dimensions
        pred_matched_idxs = self._get_src_permutation_idx(indices) # prediction idxs matched to targets

        # whl loss
        pred_3d_dims = outputs['pred_3d_dim'][pred_matched_idxs] # whl
        pred_dims_vertices, _ = generate_corners3d(pred_3d_dims, target_ry, target_3d_center_cxcy, target_depth)

        loss_dims = F.l1_loss(target_corners3d, pred_dims_vertices, reduction='none').sum()
        loss_dims /= num_boxes

        return {'cbr_loss_dims': loss_dims}
    
    def cbr_loss_ry(self, outputs:dict, targets:dict, indices, num_boxes,**kwargs)->dict:
        # Constructing Gt 3d corners
        target_3d_center_cxcy = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_3d_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0) # makes a single tensor for all targets in the batch
        target_ry = torch.cat([t['ry'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_img_size = torch.cat([t['img_size_repeat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_calib_P2 = torch.cat([t['calibs'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # _train_root_path = '/home/iony/DTU/s24/adlcv/project/adlcv_project/models/monoDETR/data/KITTIDataset_reduced/training/'
        # target_calibs = viz_utils.get_calibs(_train_root_path, targets_info['img_id'])

        # img to camera coordinates for all target (matrix form)
        target_cucv = target_3d_center_cxcy*target_img_size
        _cus = target_calib_P2[:,0,2]
        _cvs = target_calib_P2[:,1,2]
        _fus = target_calib_P2[:,0,0]
        _fvs = target_calib_P2[:,1,2]
        _txs = target_calib_P2[:,0,3] / (-_fus)
        _tys = target_calib_P2[:,1,3] / (-_fvs)

        _xs3d = (target_cucv[:,0]-_cus).unsqueeze(-1) * target_depth / _fus.unsqueeze(-1) + _txs.unsqueeze(-1)
        _ys3d = (target_cucv[:,1]-_cvs).unsqueeze(-1) * target_depth / _fvs.unsqueeze(-1) + _tys.unsqueeze(-1)
        _xys3d = torch.stack((_xs3d, _ys3d),dim=1).squeeze()
        
        target_corners3d, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, target_depth)

        # predicted 3d dimensions
        pred_matched_idxs = self._get_src_permutation_idx(indices) # prediction idxs matched to targets
        
        # ry loss
        pred_heading = outputs['pred_angle'][pred_matched_idxs] # size 24: 12 for classification, 12 for residual
        # adapted from kitti_utils.Calibration
        _fu = target_calib_P2[:, 0, 0]
        _fv = target_calib_P2[:, 1, 1]
        _cu = target_calib_P2[:, 0, 2]
        _cv = target_calib_P2[:, 1, 2]
        _tx = target_calib_P2[:, 0, 3] / _fu
        _ty = target_calib_P2[:, 1, 3] / _fv

        # scale normalized u,v to image size
        pred_out_bbox = outputs['pred_boxes'][pred_matched_idxs] # cxcylrtb
        _u = pred_out_bbox[:, 0] * target_img_size[:,0]
        _v = pred_out_bbox[:, 1] * target_img_size[:,1]

        if pred_heading.shape[0] != 0: # if there are any predictions-target matches
            # rotation loss
            alpha = torch.cat([i.reshape(1,) for i in map(get_heading_angle, pred_heading)], dim=0)
            pred_ry = torch.cat([i.reshape(1,) for i in map(alpha2ry, alpha, _u, _cu, _fu)], dim=0)
            pred_ry_vertices, U = generate_corners3d(target_3d_dims, pred_ry, target_3d_center_cxcy, target_depth)

            loss_ry = chamfer_loss(target_corners3d, pred_ry_vertices).sum()
            loss_ry /= num_boxes
        else:
            loss_ry = 0

        return {'cbr_loss_ry': loss_ry}

    def cbr_loss_all(self, outputs:dict, targets:dict, indices, num_boxes,**kwargs)->dict:
        # Constructing Gt 3d corners
        target_3d_center_cxcy = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_3d_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0) # makes a single tensor for all targets in the batch
        target_ry = torch.cat([t['ry'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_img_size = torch.cat([t['img_size_repeat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_calib_P2 = torch.cat([t['calibs'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # _train_root_path = '/home/iony/DTU/s24/adlcv/project/adlcv_project/models/monoDETR/data/KITTIDataset_reduced/training/'
        # target_calibs = viz_utils.get_calibs(_train_root_path, targets_info['img_id'])

        # img to camera coordinates for all target (matrix form)
        target_cucv = target_3d_center_cxcy*target_img_size
        _cus = target_calib_P2[:,0,2]
        _cvs = target_calib_P2[:,1,2]
        _fus = target_calib_P2[:,0,0]
        _fvs = target_calib_P2[:,1,2]
        _txs = target_calib_P2[:,0,3] / (-_fus)
        _tys = target_calib_P2[:,1,3] / (-_fvs)

        _xs3d = (target_cucv[:,0]-_cus).unsqueeze(-1) * target_depth / _fus.unsqueeze(-1) + _txs.unsqueeze(-1)
        _ys3d = (target_cucv[:,1]-_cvs).unsqueeze(-1) * target_depth / _fvs.unsqueeze(-1) + _tys.unsqueeze(-1)
        _xys3d = torch.stack((_xs3d, _ys3d),dim=1).squeeze()
        
        target_corners3d, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, target_depth)


        # predictions
        pred_matched_idxs = self._get_src_permutation_idx(indices) # prediction idxs matched to targets

        pred_depth = outputs['pred_depth'][:, :, 0][pred_matched_idxs].unsqueeze(dim=1)
        pred_out_bbox = outputs['pred_boxes'][pred_matched_idxs] # cxcylrtb
        pred_3d_dims = outputs['pred_3d_dim'][pred_matched_idxs]
        pred_heading = outputs['pred_angle'][pred_matched_idxs] # size 24: 12 for classification, 12 for residual

        # projeting prediction cucv to xy3D | adapted from kitti_utils.Calibration
        _fu = target_calib_P2[:, 0, 0]
        _fv = target_calib_P2[:, 1, 1]
        _cu = target_calib_P2[:, 0, 2]
        _cv = target_calib_P2[:, 1, 2]
        _tx = target_calib_P2[:, 0, 3] / _fu
        _ty = target_calib_P2[:, 1, 3] / _fv

        # scale normalized u,v to image size
        _u = pred_out_bbox[:, 0] * target_img_size[:,0]
        _v = pred_out_bbox[:, 1] * target_img_size[:,1]

        # project to 3d (camera coordinates)
        _x3D = ((_u-_cu)*target_depth.squeeze() / _fu ) + _tx
        _y3D = ((_v-_cv)*target_depth.squeeze() / _fv) + _ty

        pred_3d_center_xy = torch.cat((_x3D.reshape(-1, 1), _y3D.reshape(-1, 1)), dim=1)

        if pred_heading.shape[0] != 0: # if there are any predictions-target matches
            # rotation loss
            alpha = torch.cat([i.reshape(1,) for i in map(get_heading_angle, pred_heading)], dim=0)
            pred_ry = torch.cat([i.reshape(1,) for i in map(alpha2ry, alpha, _u, _cu, _fu)], dim=0)

            # entangled ('all') loss
            pred_vertices, _ = generate_corners3d(pred_3d_dims, pred_ry, pred_3d_center_xy, pred_depth)
            loss_all = chamfer_loss(target_corners3d, pred_vertices).sum()

            loss_all /= num_boxes
        else:
            loss_all = torch.tensor([0]).to(pred_depth.device)  #QUESTION: This looses the backtracking, is it a problem?
        return {'cbr_loss_all': loss_all}

    def cbr_loss_3D(self, outputs:dict, targets:dict, indices, num_boxes,**kwargs)->dict:
        # indices: tuple of 'batch_size' length with a tuple of the indices of the matched pred-targets in each batch
        # target_idx = indices[b][1]
        # Construct cubes (bs, 8, 3): GT cube and predicted (disentangled) cube
        # bs: batch size

        # target's variables, matching the size of the batch indices

        # Constructing Gt 3d corners
        target_3d_center_cxcy = torch.cat([t['boxes_3d'][:, 0: 2][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_depth = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_3d_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0) # makes a single tensor for all targets in the batch
        target_ry = torch.cat([t['ry'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_img_size = torch.cat([t['img_size_repeat'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_calib_P2 = torch.cat([t['calibs'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # _train_root_path = '/home/iony/DTU/s24/adlcv/project/adlcv_project/models/monoDETR/data/KITTIDataset_reduced/training/'
        # target_calibs = viz_utils.get_calibs(_train_root_path, targets_info['img_id'])

        # img to camera coordinates for all target (matrix form)
        target_cucv = target_3d_center_cxcy*target_img_size
        _cus = target_calib_P2[:,0,2]
        _cvs = target_calib_P2[:,1,2]
        _fus = target_calib_P2[:,0,0]
        _fvs = target_calib_P2[:,1,2]
        _txs = target_calib_P2[:,0,3] / (-_fus)
        _tys = target_calib_P2[:,1,3] / (-_fvs)

        _xs3d = (target_cucv[:,0]-_cus).unsqueeze(-1) * target_depth / _fus.unsqueeze(-1) + _txs.unsqueeze(-1)
        _ys3d = (target_cucv[:,1]-_cvs).unsqueeze(-1) * target_depth / _fvs.unsqueeze(-1) + _tys.unsqueeze(-1)
        _xys3d = torch.stack((_xs3d, _ys3d),dim=1).squeeze()
        
        target_corners3d, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, target_depth)

        # predicted 3d dimensions
        pred_matched_idxs = self._get_src_permutation_idx(indices) # prediction idxs matched to targets

        # z loss
        pred_depth = outputs['pred_depth'][:, :, 0][pred_matched_idxs].unsqueeze(dim=1)
        pred_depth_vertices, _ = generate_corners3d(target_3d_dims, target_ry, _xys3d, pred_depth)

        loss_depth = F.l1_loss(target_corners3d, pred_depth_vertices, reduction='none').sum()
        loss_depth /= num_boxes

        # u,v loss
        pred_out_bbox = outputs['pred_boxes'][pred_matched_idxs] # cxcylrtb

        # adapted from kitti_utils.Calibration
        _fu = target_calib_P2[:, 0, 0]
        _fv = target_calib_P2[:, 1, 1]
        _cu = target_calib_P2[:, 0, 2]
        _cv = target_calib_P2[:, 1, 2]
        _tx = target_calib_P2[:, 0, 3] / _fu
        _ty = target_calib_P2[:, 1, 3] / _fv

        # scale normalized u,v to image size
        _u = pred_out_bbox[:, 0] * target_img_size[:,0]
        _v = pred_out_bbox[:, 1] * target_img_size[:,1]

        # project to 3d (camera coordinates)
        _x3D = ((_u-_cu)*target_depth.squeeze() / _fu ) + _tx
        _y3D = ((_v-_cv)*target_depth.squeeze() / _fv) + _ty

        pred_3d_center_xy = torch.cat((_x3D.reshape(-1, 1), _y3D.reshape(-1, 1)), dim=1)
        pred_xy_vertices, _ = generate_corners3d(target_3d_dims, target_ry, pred_3d_center_xy, target_depth)

        loss_xy = F.l1_loss(target_corners3d, pred_xy_vertices, reduction='none').sum()
        loss_xy /= num_boxes

        # whl loss
        pred_3d_dims = outputs['pred_3d_dim'][pred_matched_idxs] # whl
        pred_dims_vertices, _ = generate_corners3d(pred_3d_dims, target_ry, target_3d_center_cxcy, target_depth)

        loss_dims = F.l1_loss(target_corners3d, pred_dims_vertices, reduction='none').sum()
        loss_dims /= num_boxes

        # ry loss
        pred_heading = outputs['pred_angle'][pred_matched_idxs] # size 24: 12 for classification, 12 for residual

        if pred_heading.shape[0] != 0: # if there are any predictions-target matches
            # rotation loss
            alpha = torch.cat([i.reshape(1,) for i in map(get_heading_angle, pred_heading)], dim=0)
            pred_ry = torch.cat([i.reshape(1,) for i in map(alpha2ry, alpha, _u, _cu, _fu)], dim=0)
            pred_ry_vertices, U = generate_corners3d(target_3d_dims, pred_ry, target_3d_center_cxcy, target_depth)

            loss_ry = chamfer_loss(target_corners3d, pred_ry_vertices).sum()
            loss_ry /= num_boxes

            # entangled ('all') loss
            pred_vertices, _ = generate_corners3d(pred_3d_dims, pred_ry, pred_3d_center_xy, pred_depth)
            loss_all = chamfer_loss(target_corners3d, pred_vertices).sum()
            loss_all /= num_boxes
        else:
            loss_ry = 0
            loss_all = 0

        # sigma = outputs['pred_depth'][:, :, 1: 2]
        # sigma = torch.exp(-sigma) #QUESTION: What is this sigma?

        losses = {}
        losses['cbr_loss_3D'] = loss_depth + loss_xy + loss_dims + loss_ry + loss_all

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(cfg):
    # backbone
    backbone = build_backbone(cfg)

    # detr
    depthaware_transformer = build_depthaware_transformer(cfg)

    # depth prediction module
    depth_predictor = DepthPredictor(cfg)

    model = MonoDETR(
        backbone,
        depthaware_transformer,
        depth_predictor,
        num_classes=cfg['num_classes'],
        num_queries=cfg['num_queries'],
        aux_loss=cfg['aux_loss'],
        num_feature_levels=cfg['num_feature_levels'],
        with_box_refine=cfg['with_box_refine'],
        two_stage=cfg['two_stage'],
        init_box=cfg['init_box'],
        use_dab = cfg['use_dab'],
        group_num = cfg['group_num'],
        two_stage_dino=cfg['two_stage_dino'],
        depth_predictor2loss=cfg['depth_predictor2loss'])

    # matcher
    matcher = build_matcher(cfg)

    # loss
    weight_dict = {'loss_ce': cfg['cls_loss_coef'], 'loss_bbox': cfg['bbox_loss_coef']}
    weight_dict['loss_giou'] = cfg['giou_loss_coef']
    weight_dict['loss_dim'] = cfg['dim_loss_coef']
    weight_dict['loss_angle'] = cfg['angle_loss_coef']
    weight_dict['loss_depth'] = cfg['depth_loss_coef']
    weight_dict['loss_center'] = cfg['3dcenter_loss_coef']
    weight_dict['loss_depth_map'] = cfg['depth_map_loss_coef']

    # added cube-rcnn losses    
    weight_dict['cbr_loss_depth']   = cfg['cbr_loss_depth_coef']
    weight_dict['cbr_loss_xy']      = cfg['cbr_loss_xy_coef']
    weight_dict['cbr_loss_dims']    = cfg['cbr_loss_dims_coef']
    weight_dict['cbr_loss_ry']      = cfg['cbr_loss_ry_coef']
    weight_dict['cbr_loss_all']     = cfg['cbr_loss_all_coef']


    # dn loss
    if cfg['use_dn']:
        weight_dict['tgt_loss_ce']= cfg['cls_loss_coef']
        weight_dict['tgt_loss_bbox'] = cfg['bbox_loss_coef']
        weight_dict['tgt_loss_giou'] = cfg['giou_loss_coef']
        weight_dict['tgt_loss_angle'] = cfg['angle_loss_coef']
        weight_dict['tgt_loss_center'] = cfg['3dcenter_loss_coef']

    # TODO this is a hack
    if cfg['aux_loss']:
        aux_weight_dict = {}
        for i in range(cfg['dec_layers'] - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    applied_losses = ['labels',
    'boxes',
    'cardinality',
    'depths',
    'dims',
    'angles',
    'center',
    'depth_map']

    applied_losses += [
        'cbr_loss_depth',
        'cbr_loss_xy',
        'cbr_loss_dims',
        # 'cbr_loss_ry',
        'cbr_loss_all'
    ]

    criterion = ExtendedSetCriterion(
        cfg['num_classes'],
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=cfg['focal_alpha'],
        applied_losses=applied_losses)

    device = torch.device(cfg['device'])
    criterion.to(device)

    return model, criterion
