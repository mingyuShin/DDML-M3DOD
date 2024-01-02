import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.helpers.decode_helper import _transpose_and_gather_feat
from lib.losses.focal_loss import focal_loss_cornernet
from lib.losses.uncertainty_loss import laplacian_aleatoric_uncertainty_loss
from lib.losses.dim_aware_loss import dim_aware_l1_loss


def _update_K_return_loss(feat, dep_tar, K, B, rad, temperature=10.0):
    lambda_metric = 0.5
    loss = 0.0
    x = feat.unsqueeze(0) # 1xnum_obj_cls x C
    feat_dist = torch.cdist(x, x.detach(), p=2.0).squeeze(0) # num_obj_cls x num_obj_cls    e_ij : L2 distance between features of obj_i & obj_2
    y = dep_tar.unsqueeze(0)
    dep_dist = torch.cdist(y, y, p=2.0).squeeze(0) # num_obj_cls x num_obj_cls    e_ij : L2 distance between depths of obj_i & obj_2
    upper_slope = torch.triu((feat_dist - (K * dep_dist) - B), diagonal=1) # positive values in this matrix is out of upper bound
    lower_slope = torch.triu(((1.0/K) * dep_dist - feat_dist - B), diagonal=1) # positive values in this matrix is out of lower bound
    mask = (dep_dist <= rad) * (dep_dist > 1e-12)
    masked_upper_slope = upper_slope[mask]
    masked_lower_slope = lower_slope[mask]

    pos_ancs = torch.exp(-masked_upper_slope[masked_upper_slope > 0] / temperature) # N_obj
    neg_anc = torch.sum(torch.exp(-masked_lower_slope[masked_lower_slope > 0] / temperature)) # scalar
    tmp = torch.mean(-torch.log(pos_ancs / (pos_ancs + neg_anc))) # R^N / R^N -> R^N vector
    loss += torch.nan_to_num(tmp, nan=0.0, posinf=0.0, neginf=0.0)

    return lambda_metric * loss

def compute_centernet3d_loss(input, target, feat, K, B, rad, qi, obj):
    stats_dict = {}

    seg_loss = compute_segmentation_loss(input, target)
    offset2d_loss = compute_offset2d_loss(input, target)
    size2d_loss = compute_size2d_loss(input, target)
    offset3d_loss = compute_offset3d_loss(input, target)
    depth_loss = compute_depth_loss(input, target)
    size3d_loss = compute_size3d_loss(input, target)
    heading_loss = compute_heading_loss(input, target)

    # statistics
    stats_dict['seg'] = seg_loss.item()
    stats_dict['offset2d'] = offset2d_loss.item()
    stats_dict['size2d'] = size2d_loss.item()
    stats_dict['offset3d'] = offset3d_loss.item()
    stats_dict['depth'] = depth_loss.item()
    stats_dict['size3d'] = size3d_loss.item()
    stats_dict['heading'] = heading_loss.item()
    
    obj_loss = 0.
    if obj:
        obj_loss += torch.nan_to_num(compute_obj_depth_loss(input, target, feat), nan=0.0, posinf=0.0, neginf=0.0)

    qi_loss = 0.
    if qi:
        ## avg pooling
        feat_avg = F.avg_pool2d(feat, kernel_size=(5,5), stride=1, padding=2)
        x = extract_input_from_tensor(feat_avg, target['indices'], target['mask_3d'])
        
        dep_tar = extract_target_from_tensor(target['depth'], target['mask_3d'])
        qi_loss += _update_K_return_loss(x, dep_tar, K, B, rad)

    total_loss = seg_loss + offset2d_loss + size2d_loss + offset3d_loss + \
                 depth_loss + size3d_loss + heading_loss + qi_loss + obj_loss
    return total_loss, stats_dict


def compute_obj_depth_loss(input, target, feat):
    dep_in = input['obj_depth']
    dep_tar = F.interpolate(target['obj_depth'].unsqueeze(1), size=(feat.shape[-2],feat.shape[-1]), mode='bilinear', align_corners=True)
    tar_mask = ~(dep_tar == 0)
    depth_input, depth_log_variance = dep_in[:,0:1][tar_mask], dep_in[:,1:2][tar_mask]
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_target = dep_tar[tar_mask]
    depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)     
    return depth_loss

def compute_segmentation_loss(input, target):
    input['heatmap'] = torch.clamp(input['heatmap'].sigmoid_(), min=1e-4, max=1 - 1e-4)
    loss = focal_loss_cornernet(input['heatmap'], target['heatmap'])
    return loss


def compute_size2d_loss(input, target):
    # compute size2d loss
    size2d_input = extract_input_from_tensor(input['size_2d'], target['indices'], target['mask_2d'])
    size2d_target = extract_target_from_tensor(target['size_2d'], target['mask_2d'])
    size2d_loss = F.l1_loss(size2d_input, size2d_target, reduction='mean')
    return size2d_loss

def compute_offset2d_loss(input, target):
    # compute offset2d loss
    offset2d_input = extract_input_from_tensor(input['offset_2d'], target['indices'], target['mask_2d'])
    offset2d_target = extract_target_from_tensor(target['offset_2d'], target['mask_2d'])
    offset2d_loss = F.l1_loss(offset2d_input, offset2d_target, reduction='mean')
    return offset2d_loss


def compute_depth_loss(input, target):
    depth_input = extract_input_from_tensor(input['depth'], target['indices'], target['mask_3d'])
    depth_input, depth_log_variance = depth_input[:, 0:1], depth_input[:, 1:2]
    depth_input = 1. / (depth_input.sigmoid() + 1e-6) - 1.
    depth_target = extract_target_from_tensor(target['depth'], target['mask_3d'])
    depth_loss = laplacian_aleatoric_uncertainty_loss(depth_input, depth_target, depth_log_variance)
    return depth_loss


def compute_offset3d_loss(input, target):
    offset3d_input = extract_input_from_tensor(input['offset_3d'], target['indices'], target['mask_3d'])
    offset3d_target = extract_target_from_tensor(target['offset_3d'], target['mask_3d'])
    offset3d_loss = F.l1_loss(offset3d_input, offset3d_target, reduction='mean')
    return offset3d_loss


def compute_size3d_loss(input, target):
    size3d_input = extract_input_from_tensor(input['size_3d'], target['indices'], target['mask_3d'])
    size3d_target = extract_target_from_tensor(target['size_3d'], target['mask_3d'])
    size3d_loss = dim_aware_l1_loss(size3d_input, size3d_target, size3d_target)
    return size3d_loss


def compute_heading_loss(input, target):
    heading_input = _transpose_and_gather_feat(input['heading'], target['indices'])   # B * C * H * W ---> B * K * C
    heading_input = heading_input.view(-1, 24)
    heading_target_cls = target['heading_bin'].view(-1)
    heading_target_res = target['heading_res'].view(-1)
    mask = target['mask_2d'].view(-1)

    # classification loss
    heading_input_cls = heading_input[:, 0:12]
    heading_input_cls, heading_target_cls = heading_input_cls[mask], heading_target_cls[mask]
    if mask.sum() > 0:
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='mean')
    else:
        cls_loss = 0.0

    # regression loss
    heading_input_res = heading_input[:, 12:24]
    heading_input_res, heading_target_res = heading_input_res[mask], heading_target_res[mask]
    cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
    heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
    reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='mean')
    return cls_loss + reg_loss


######################  auxiliary functions #########################

def extract_input_from_tensor(input, ind, mask):
    input = _transpose_and_gather_feat(input, ind)  # B*C*H*W --> B*K*C
    return input[mask]  # B*K*C --> M * C

def extract_target_from_tensor(target, mask):
    return target[mask]


if __name__ == '__main__':
    input_cls  = torch.zeros(2, 50, 12)  # B * 50 * 24
    input_reg  = torch.zeros(2, 50, 12)  # B * 50 * 24
    target_cls = torch.zeros(2, 50, 1, dtype=torch.int64)
    target_reg = torch.zeros(2, 50, 1)

    input_cls, target_cls = input_cls.view(-1, 12), target_cls.view(-1)
    cls_loss = F.cross_entropy(input_cls, target_cls, reduction='mean')

    a = torch.zeros(2, 24, 10, 10)
    b = torch.zeros(2, 10).long()
    c = torch.ones(2, 10).long()
    d = torch.zeros(2, 10, 1).long()
    e = torch.zeros(2, 10, 1)
    print(compute_heading_loss(a, b, c, d, e))

