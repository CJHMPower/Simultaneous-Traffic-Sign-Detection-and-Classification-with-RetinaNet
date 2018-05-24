from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def one_hot(index, classes):
   
    size = index.size() + (classes,)
    view = index.size() + (1,)
    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.
    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)
    return mask.scatter_(1, index, ones)

def one_hot_embedding(labels, num_classes):
    
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]            # [N,D]
    
class FocalLoss(nn.Module):
    def __init__(self, num_classes):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        y = one_hot(y.cpu(), x.size(-1)).cuda()
        logit = F.softmax(x)
        logit = logit.clamp(1e-7, 1. - 1e-7)
        loss = -1 * y.float() * torch.log(logit)
        loss =loss * (1 - logit) ** 2
        return loss.sum()
    
        
        
    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets,pos,verbose):
        batch_size, num_boxes = cls_targets.size()
        num_pos = pos.data.long().sum()
        mask = pos.unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask].view(-1,4)
        masked_loc_targets = loc_targets[mask].view(-1,4)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets,size_average=False)      
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes+1)
        cls_loss =self.focal_loss(masked_cls_preds, cls_targets[pos_neg])
        if verbose:
            print('loc_loss: %.5f | cls_loss: %.4f' %((loc_loss.data[0]/num_pos), cls_loss.data[0]/(num_pos)), end=' | ')

        loss =(loc_loss+cls_loss)/(num_pos)
        return loss