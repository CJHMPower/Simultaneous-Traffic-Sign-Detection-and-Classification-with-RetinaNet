import os
import sys
import time
import math

import torch
import torch.nn as nn
import numpy as np

def get_mean_and_std(dataset, max_load=100000):
    '''Compute the mean and std value of dataset.'''
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    N = min(max_load, len(dataset))
    for i in range(N):
        im,_,_ = dataset.load(1)
        for j in range(3):
            mean[j] += im[:,j,:,:].mean()
            std[j] += im[:,j,:,:].std()
    mean.div_(N)
    std.div_(N)
    return mean, std

def meshgrid(x, y, swap_dims=False):
    '''Return meshgrid in range x & y.

    Args:
      x: (int) first dim range.
      y: (int) second dim range.
      swap_dims: (bool) swap dims.

    Returns:
      (tensor) meshgrid, sized [x*y,2]

    Example:
    >> meshgrid(3,2)
    0  0
    0  1
    1  0
    1  1
    2  0
    2  1
    [torch.FloatTensor of size 6x2]
    '''
    a = torch.arange(0,x)
    b = torch.arange(0,y)
    xx = a.view(-1,1).repeat(1,y).view(-1,1)
    yy = b.repeat(x,1).view(-1,1)
    return torch.cat([yy,xx],1) if swap_dims else torch.cat([xx,yy],1)

def change_box_order(boxes, order):
    assert order in ['xyxy2xywh','xywh2xyxy']
    a = boxes[:,:2]
    b = boxes[:,2:]
    if order == 'xyxy2xywh':
        return torch.cat([(a+b)/2,b-a], 1)
    return torch.cat([a-b/2,a+b/2], 1)

def box_iou(box1, box2, order='xyxy'):
    if order == 'xywh':
        box1 = change_box_order(box1, 'xywh2xyxy')
        box2 = change_box_order(box2, 'xywh2xyxy')

    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = (rb-lt).clamp(min=0)  # [N,M,2]
    inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou

def box_nms(bboxes, scores, threshold=0.5):
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h
        ovr = inter / areas[order[1:]].clamp(max=areas[i])
        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)
