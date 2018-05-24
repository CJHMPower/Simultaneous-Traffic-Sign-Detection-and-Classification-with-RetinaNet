import numpy as np
import os
import cv2
from PIL import Image
from utils import box_iou, box_nms, change_box_order, meshgrid
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import preprocessing.transforms as transforms
from encoder import DataEncoder
from loss import *
from retinanet import RetinaNet
from preprocessing.datasets import VocLikeDataset
import matplotlib.pyplot as plt
import config as cfg
from tqdm import tqdm
import evaluate.anno_func
import json
from optparse import OptionParser


def load_model(backbone):
    print('loading model...')
    model= torch.load(os.path.join('ckpts', 'model',backbone+'_retinanet.pth'))
    net=RetinaNet(backbone=backbone,num_classes=len(cfg.classes))
    net=torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.cuda()
    cudnn.benchmark = True
    net.load_state_dict(model['net'])
    return net

def vis(img, boxes, labels, classes,color):
    img=img.copy()
    for box,label in zip(boxes,labels):
        cv2.rectangle(img, (max(0,int(box[0])),max(0,int(box[1]))),(min(511,int(box[2])),min(511,int(box[3]))),color,2)
        ss=cfg.classes[label-1]
        cv2.putText(img, ss, (int(box[0]),int(box[1]-10)), 0, 0.6, color, 2)
    return img


def eval_valid(net, valloader, anno_file,image_dir):
    net.eval()
    annos_pred={}
    annos_pred['imgs']={}
    annos=json.loads(open(anno_file).read())
    annos_target={}
    annos_target['imgs']={}
    
    for batch_idx, (inputs, loc_targets, cls_targets) in tqdm(enumerate(valloader)):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())
        loc_preds, cls_preds = net(inputs)
        for i in range(loc_preds.size()[0]):
            imgid=cfg.val_imageset_fn[batch_idx*batch_size+i].split('/')[-1][:-4]
            annos_target['imgs'][imgid]=annos['imgs'][imgid]
            boxes,labels,score=DataEncoder().decode(loc_preds[i], cls_preds[i], input_size=512)
            annos_pred['imgs'][imgid]={}
            rpath=os.path.join(image_dir,imgid+'.jpg')
            annos_pred['imgs'][imgid]['path']=rpath
            annos_pred['imgs'][imgid]['objects']=[]
            if boxes is None:
                continue
            for i,box in enumerate(boxes):
                bbox={}
                bbox['xmin']=box[0]
                bbox['xmax']=box[2]
                bbox['ymin']=box[1]
                bbox['ymax']=box[3]

            
                annos_pred['imgs'][imgid]['objects'].append({'score':100*float(score[i]),'bbox':bbox,'category':cfg.classes[labels[i]-1]})
    
    print('Test done, evaluating result...')
    
    
    with open(os.path.join(datadir,predict_dir),'w') as f:
        json_str=json.dumps(annos_pred)
        json.dump(annos_pred,f)
        f.close()
    with open(os.path.join(datadir,target_dir),'w') as f:
        json_str=json.dumps(annos_target)
        json.dump(annos_target,f)
        f.close()
    
def test_image(net, imgid_path,file_name):
    img=Image.open(os.path.join(imgid_path,file_name))
    width, height=img.size
    if width!=cfg.width or height!=cfg.height:
        img=cv2.resize(img,(cfg.width, cfg.height))
    img=np.asarray(img)
    image = img.transpose((2, 0, 1))
    image=torch.from_numpy(image)
    if isinstance(image, torch.ByteTensor):
        image = image.float().div(255)
    for t, m, s in zip(image, cfg.mean, cfg.std):
        t.sub_(m).div_(s)
    net.eval()
    image=Variable(image.resize_(1,3,cfg.width,cfg.height))
    loc_pred, cls_pred=net(image)
    boxes,labels,score=DataEncoder().decode(loc_pred[0], cls_pred[0], input_size=(cfg.width,cfg.height))
    if boxes is None:
        new_img=img
    else:
        new_img=vis(img, boxes, labels, cfg.classes, (0,0,255))
    return new_img
    
    
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--mode', dest='mode',default='demo',
		help='Operating mode, could be demo or valid, demo mode will provide visulization results for images in samples/')
    
    parser.add_option('--backbone','--backbone',dest='backbone',default='resnet101',
    help='Backbone pretrained model, could be resnet50, resnet101 or resnet152')
    
    options, args = parser.parse_args()
    mode = options.mode
    backbone=options.backbone
    if backbone not in ['resnet50', 'resnet101', 'resnet152']:
        assert ValueError('Invalid backbone: %s' % backbone)
    net=load_model(backbone)
    if mode=='valid':
        datadir=cfg.root
        batch_size=2
        anno_file=os.path.join(datadir,'annotation.json')
        target_dir='valid_target.json'
        predict_dir=backbone+'_predict.json'
        val_transform = transforms.Compose([
        transforms.ToTensor(),transforms.Normalize(cfg.mean, cfg.std)])
        valset = VocLikeDataset(image_dir=cfg.val_image_dir, annotation_file=cfg.annotation_file, imageset_fn=cfg.val_imageset_fn,
                        image_ext=cfg.image_ext, classes=cfg.classes, encoder=DataEncoder(), transform=val_transform)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                        num_workers=cfg.num_workers, collate_fn=valset.collate_fn)
        eval_valid(net, valloader,anno_file, cfg.test_dir)
        filedir=os.path.join(datadir, target_dir)
        annos = json.loads(open(filedir).read())
        result_anno_file=os.path.join(datadir,predict_dir)
        results_annos1 = json.loads(open(result_anno_file).read())
        print (len(results_annos1['imgs']))
        sm = anno_func.eval_annos(annos, results_annos1, iou=0.5,types=anno_func.type42,minscore=50,check_type=True)
        print sm['report']
        
    elif mode=='demo':
        image_dir='samples'
        img_list=os.listdir(image_dir)
        for fname in img_list:
            new_img=test_image(net, image_dir, fname)
            plt.imshow(new_img)
            plt.show()
    else:
        assert ValueError('Invalid mode: %s' % mode)
        
    
