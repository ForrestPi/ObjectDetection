# -*- coding: utf-8 -*-
import sys
sys.path.insert(0,"/data/yiling/code/detect/FCOS.pytorch/models")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import one_hot_embedding

INF = 999999


class IOU_loss(nn.Module,): 
    def forward(self, pred, target,weight=None):
        
        #pred
        left = pred[:,0]
        top = pred[:,1]
        right = pred[:,2]
        bottom = pred[:,3]
        # ground truth
        left_hat = target[:,0]
        top_hat = target[:,1]
        right_hat = target[:,2]
        bottom_hat = target[:,3]
        #compute iou area
        pred_area = (left+right) * (top+bottom)
        target_area = (left_hat+right_hat)*(top_hat+bottom_hat)
        #compute box iou
        inter_h = torch.min(bottom, bottom_hat) + torch.min(top,top_hat)
        inter_w = torch.min(left, left_hat) + torch.min(right,right_hat)
        
        intersection = inter_h * inter_w
        union = pred_area + target_area - intersection
        loss = -torch.log((intersection+1.) / (union+1))
        
        if weight is not None:
            return (loss*weight).sum() / weight.sum()
        else:
            return loss.mean()
        
class focal_loss(nn.Module):
    def __init__(self, alpha, gamma):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, x, y, classes=20):
        t = one_hot_embedding(y, classes+1)  # classes include background
        t = t[:,1:]                          # exclude background
        p = x.sigmoid()
        pt = torch.where(t>0, p, 1-p)    # pt = p if t > 0 else 1-p
        
        w = (1-pt).pow(self.gamma)
        w = torch.where(t>0, self.alpha*w, (1-self.alpha)*w).detach()    
        loss = F.binary_cross_entropy_with_logits(x, t, w, reduction='sum')
  
        return loss
        
        

class mul_task_loss(object):
    def __init__(self, cfg=None):
            
#        self.cls_criteron = focal_loss()
        self.box_criteron = IOU_loss()
        self.cls_criteron = focal_loss(cfg['alpha'], cfg['gamma'])
        self.centerness_loss = nn.BCEWithLogitsLoss()
        #self.fpn_stride = [8, 16, 32, 64, 128]
        
    def __call__(self, cls, box, centerness, locations, targets):
        """
        Args:
            input_size:
            locations (list[BoxList])
            cls (list[Tensor,])             # [b, 32, 32, 20]   / 16, 8, 4, 2
            box (list[Tensor,])             # [b, 32, 32, 4]   / 16, 8, 4, 2
            centerness (list[Tensor,])      # [b, 32, 32, 1]
            targets (Tensor_list[Tensor])   # [b, 128, 6] : --> [x,y,x,y,c,area]
        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        B = cls[0].size(0) # batch size
        N = cls[0].size(-1) #number classes
    

        labels, reg_targets = self.get_targets(locations, targets)
        
        
        cls_flatten = []
        boxes_flatten = []
        centerness_flatten = []
        
        labels_flatten = []
        reg_targets_flatten = []
        
        for idx in range(len(labels)):
            cls_flatten.append(cls[idx].reshape(-1,N))              
            boxes_flatten.append(box[idx].reshape(-1, 4))            
            centerness_flatten.append(centerness[idx].reshape(-1))   
            
            labels_flatten.append(labels[idx].reshape(-1))           
            reg_targets_flatten.append(reg_targets[idx].reshape(-1,4)) 
        
        # outputs
        cls_flatten = torch.cat(cls_flatten, 0)       
        boxes_flatten = torch.cat(boxes_flatten, 0)   
        centerness_flatten = torch.cat(centerness_flatten, 0) 
        # targets
        labels_flatten = torch.cat(labels_flatten, 0) 
        
        reg_targets_flatten = torch.cat(reg_targets_flatten, 0) 
        
        pos_idx = torch.nonzero(labels_flatten > 0).squeeze(1) 
        
        ## compute class loss 
        cls_loss = self.cls_criteron(cls_flatten, labels_flatten.long(), classes=N) / (pos_idx.numel())
        
        ## compute box loss
        boxes_flatten = boxes_flatten[pos_idx]
        reg_targets_flatten = reg_targets_flatten[pos_idx]
        centerness_flatten = centerness_flatten[pos_idx]

        if pos_idx.numel() > 0:
            centerness_target = self.compute_centerness_targets(reg_targets_flatten)
            box_loss = self.box_criteron(boxes_flatten, reg_targets_flatten, centerness_target)
            centerness_loss = self.centerness_loss(centerness_flatten, centerness_target)
           
        else:
            box_loss = boxes_flatten.sum()
            centerness_loss = centerness_flatten.sum()
        losses = {
                'cls_loss':cls_loss,
                'box_loss':box_loss,
                'centerness_loss':centerness_loss, 
                }
        return losses
     
    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        #
        return torch.sqrt(centerness)
    
    
    def get_targets(self, locations, targets):
        each_map_restrict = [
                 [-1, 32],
                [32, 64],
                [64, 128],
                [128, 256],
                [256, INF],
                ]
        sizes_of_interests = []
        for idx, loc in enumerate(locations):
            size_of_interst = loc.new_tensor(each_map_restrict[idx])
            sizes_of_interests.append(size_of_interst.expand_as(loc))
            
        sizes_of_interests = torch.cat(sizes_of_interests, 0) 
       
        num_locations = [len(i) for i in locations] 
        total_locations = torch.cat(locations, 0)
        
        labels, reg_targets = self.compute_targets_of_locations(total_locations, targets, sizes_of_interests)

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_locations, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_locations, dim=0)
            
        
        labels_level_first = []
        reg_targets_level_first = []
        #print([i.shape for i in locations])
        
        for level in range(len(locations)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0) 
            )
            reg_targets_level_first.append(
                    torch.cat([reg_targets_per_im[level] for reg_targets_per_im in reg_targets], dim=0)
                )
        
        return labels_level_first, reg_targets_level_first
    
    def compute_targets_of_locations(self, locations, targets, sizes_of_interests):
 
        labels = []
        reg_targets = []
        
        xs, ys = locations[:, 0], locations[:, 1]
        for i in range(len(targets)):
            target = targets[i].float()
            bboxes = target[:,:4] 
            labels_per_img = target[:,4]
            areas = target[:,-1]
            
            for idx, area in enumerate(areas):
                if area == 0.:
                    break
           
            bboxes = bboxes[:idx, :] 
            labels_per_img = labels_per_img[:idx]    
            areas = areas[:idx]      
            
            l = xs[:,None] - bboxes[:,0][None]   
            t = ys[:, None] - bboxes[:, 1][None] 
            r = bboxes[:, 2][None] - xs[:, None] 
            b = bboxes[:, 3][None] - ys[:, None] 
            
            reg_targets_per_img = torch.stack([l, t, r, b], dim=2)
            is_in_boxes = reg_targets_per_img.min(dim=2)[0] > 0   
            max_reg_targets_per_img = reg_targets_per_img.max(dim=2)[0] 

            # restrict the regression range for each location 
            reg_range = (max_reg_targets_per_img >= sizes_of_interests[:, [0]]) & \
                        (max_reg_targets_per_img <= sizes_of_interests[:, [1]])  
            
            locations_to_gt_area = areas[None].repeat(len(locations), 1)
           
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[reg_range == 0] = INF 
            
            locations_to_min_aera, locations_to_gt_inds = locations_to_gt_area.min(dim=1) 
            
            reg_targets_per_img = reg_targets_per_img[range(len(xs)), locations_to_gt_inds] 
            labels_per_img = labels_per_img[locations_to_gt_inds] 
            labels_per_img[locations_to_min_aera == INF] = 0 
            labels.append(labels_per_img)
            reg_targets.append(reg_targets_per_img)
        return labels, reg_targets




from box_utils import bbox_overlaps_iou,bbox_overlaps_giou,\
    bbox_overlaps_giou,bbox_overlaps_diou,bbox_overlaps_ciou
class IOU_loss(nn.Module): 
    def forward(self, pred, target,weight=None,losstype='Giou'):
        
        if losstype == 'Iou':
            loss = 1.0 - bbox_overlaps_iou(pred, target)
        else:
            if losstype == 'Giou':
                loss = 1.0 - bbox_overlaps_giou(pred, target)
            else:
                if losstype == 'Diou':
                    loss = 1.0 - bbox_overlaps_diou(pred, target)
                else:
                    loss = 1.0 - bbox_overlaps_ciou(pred, target)           
        
        if weight is not None:
            return (loss*weight).sum() / weight.sum()
        else:
            return loss.mean()
        