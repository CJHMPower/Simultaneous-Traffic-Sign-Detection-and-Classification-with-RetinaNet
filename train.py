import argparse
import os
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import preprocessing.transforms as transforms
from encoder import DataEncoder
from loss import FocalLoss
from retinanet import RetinaNet
from preprocessing.datasets import VocLikeDataset


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--exp', required=True, help='experiment name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

#sys.path.insert(0, os.path.join('exps', 'voc'))
import config as cfg

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')
start_epoch = 0
lr = cfg.lr

print('Preparing data..')

train_transform_list = [transforms.ToTensor(),transforms.Normalize(cfg.mean, cfg.std)]
if cfg.scale is not None:
    train_transform_list.insert(0,transforms.Scale(cfg.scale))
train_transform = transforms.Compose(train_transform_list)
val_transform = transforms.Compose([
    transforms.ToTensor(),transforms.Normalize(cfg.mean, cfg.std)
])

trainset = VocLikeDataset(image_dir=cfg.image_dir, annotation_file=cfg.annotation_file,imageset_fn=cfg.train_imageset_fn,
                          image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=train_transform)
valset = VocLikeDataset(image_dir=cfg.image_dir, annotation_file=cfg.annotation_file, imageset_fn=cfg.val_imageset_fn,
                        image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=val_transform)

valloader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False,
                                        num_workers=cfg.num_workers, collate_fn=valset.collate_fn)

print('Building model...')
net = RetinaNet(backbone=cfg.backbone, num_classes=len(cfg.classes))
net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()
cudnn.benchmark = True

if args.resume:
    print('Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join('ckpts', args.exp, '5_ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    
    start_epoch = checkpoint['epoch']
    lr = cfg.lr




criterion = FocalLoss(len(cfg.classes))

optimizer = optim.Adam(net.parameters(), lr=cfg.lr,weight_decay=cfg.weight_decay)

def train(epoch):
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                                          num_workers=cfg.num_workers, collate_fn=trainset.collate_fn)
    print('\nTrain Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        
        optimizer.zero_grad()
        
        loc_preds, cls_preds = net(inputs)
        
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()
        if num_pos==0:
            print('zero num positive')
            continue
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets, pos,batch_idx%2==0)
        
        
        loss.backward()
        nn.utils.clip_grad_norm(net.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.data[0]
        if batch_idx%2==0:
            print('train_loss: %.3f | avg_loss: %.4f' % (loss.data[0], train_loss/(batch_idx+1)))
            
    save_checkpoint(train_loss,epoch, len(trainloader))
    
    

def val(epoch):
    net.eval()
    val_loss = 0
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        pos = cls_targets > 0
        
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets,pos,batch_idx%10==0)
        val_loss += loss.data[0]
        if batch_idx%10==0:
            print('val_loss: %.4f | avg_loss: %.4f' % (loss.data[0], val_loss/(batch_idx+1)))
            
    
    
    
    
def save_checkpoint(loss, epoch, n):
    global best_loss
    loss /= n
    if loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'loss': loss,
            'epoch': epoch,
            'lr': lr
        }
        ckpt_path = os.path.join('ckpts', args.exp)
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)
        torch.save(state, os.path.join(ckpt_path, str(epoch)+'_ckpt.pth'))
        best_loss = loss

for epoch in range(start_epoch + 1, start_epoch + cfg.num_epochs + 1):
    if epoch in cfg.lr_decay_epochs:
        lr *= 0.1
        print('learning rate decay to: ', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    train(epoch)
    if cfg.eval_while_training and epoch % cfg.eval_every == 0:
        val(epoch)
    

    
