import torch
import torch.nn.functional as F
from torch import nn

from fvcore.nn import giou_loss, smooth_l1_loss
from detectron2.layers import cat, cross_entropy, nonzero_tuple, batched_nms

from monoDETR import utils
import math


class Cube2DLoss(nn.module):
    # TODO: 

    def loss_cls():
        pass

    def loss_box_reg():
        pass



class Cube3DLoss(nn.module):

    def l1_loss(self, vals, target):
        return F.smooth_l1_loss(vals, target, reduction='none', beta=0.0)

    def chamfer_loss(self, vals, target):
        B = vals.shape[0]
        xx = vals.view(B, 8, 1, 3)
        yy = target.view(B, 1, 8, 3)
        l1_dist = (xx - yy).abs().sum(-1)
        l1 = (l1_dist.min(1).values.mean(-1) + l1_dist.min(2).values.mean(-1))
        return l1


    def calculate_gt_corners(self, gt_boxes3D, gt_poses):
        # let lowercase->2D and uppercase->3D
        # [x, y, Z, W, H, L] 
        gt_2d = gt_boxes3D[:, :2]
        gt_z = gt_boxes3D[:, 2]
        gt_dims = gt_boxes3D[:, 3:6]

        # Without considering camera intrinsics and scaling, the 3D coordinates are directly taken from the ground truth
        gt_3d = torch.cat((gt_2d, gt_z.unsqueeze(-1)), dim=1)
        
        # put together the GT boxes
        gt_box3d = torch.cat((gt_3d, gt_dims), dim=1)

        # These are the corners which will be the target for all losses!!
        gt_corners = utils.get_cuboid_verts_faces(gt_box3d, gt_poses)[0]
        return gt_3d, gt_corners
    
    def gt_angle_to_rotation_matrix(self, gt_angles):


        gt_R = 
        return gt_R


    def loss_dims(self, outputs, targets, indices, num_boxes):

        ## FROM MONODETR ##
        # Pull off predicted dimensions
        idx = self._get_src_permutation_idx(indices)
        cube_dims = outputs['pred_3d_dim'][idx] # (n, 3)

        # Pull off necessary GT information
        gt_boxes3D = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0) # TODO: how to get gt_boxes3D?

        gt_angles = torch.cat([t['angle'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # TODO: how to get gt_angles?


        ## CONVERSION BETWEEN MONODETR AND CUBE-RCNN
        gt_R = self.gt_angle_to_rotation_matrix(gt_angles)



        ## ADOPTED FROM CUBE-RCNN
        # These are the corners which will be the target for all losses!!
        gt_3d, gt_corners = self.calculate_gt_corners(gt_boxes3D, gt_poses)

        # Get the dimensions
        dis_dims_corners = utils.get_cuboid_verts_faces(torch.cat((gt_3d, cube_dims), dim=1), gt_poses)[0]

        # Calculate the loss
        loss_dims = self.l1_loss(dis_dims_corners, gt_corners)
        pass

    def loss_pose(self, outputs, targets, indices, num_boxes):
        # TODO: loss_pose
        loss_pose = self.chamfer_loss(dis_pose_corners, gt_corners)
        pass


    def loss_xy(self, outputs, targets, indices, num_boxes):
        # TODO: loss_xy
        loss_xy = self.l1_loss(dis_XY_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
        pass


    def loss_z(self, outputs, targets, indices, num_boxes):
        # TODO: loss_z

        loss_z = self.l1_loss(dis_z_corners, gt_corners).contiguous().view(n, -1).mean(dim=1)
        pass

    def loss_joint(self, outputs, targets, indices, num_boxes):
        # TODO: loss_joint
        loss_joint = self.chamfer_loss(dis_z_corners_joint, gt_corners)
        pass