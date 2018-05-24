import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data.dataset import Dataset

from preprocessing.annotations import AnnotationDir

import config as cfg

class VocLikeDataset(Dataset):
    
    def __init__(self, image_dir, annotation_file, imageset_fn, image_ext, classes, encoder, transform=None, test=False):
        self.image_dir_path = image_dir
        self.image_ext = image_ext
        self.filenames=imageset_fn
        if not test:
            self.annotation_dir = AnnotationDir(annotation_file,classes)

        self.encoder = encoder
        self.transform = transform
        self.test = test
        

    def __getitem__(self, index):
        
        image_fn = self.filenames[index]
        fn=os.path.join(self.image_dir_path.split('/')[-1],image_fn)
        image_path = os.path.join(self.image_dir_path, image_fn)
        image = Image.open(os.path.join('data', image_fn))
        example={}
        example['image']=image
        if not self.test:
            boxes = self.annotation_dir.get_boxes(image_fn.split('/')[-1])
            example['boxes']=boxes
       
        if self.transform:
            example = self.transform(example)
        return example

    def __len__(self):
        return len(self.filenames)

    def collate_fn(self, batch):
        imgs = [example['image'] for example in batch]
        if not self.test:
            boxes  = [example['boxes'] for example in batch]
            labels = [example['labels'] for example in batch]
        img_sizes = [img.size()[1:] for img in imgs]
        
        max_h = max([im.size(1) for im in imgs])
        max_w = max([im.size(2) for im in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, max_h, max_w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            im = imgs[i]
            imh, imw = im.size(1), im.size(2)
            inputs[i,:,:imh,:imw] = im
            if not self.test:
                loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=[cfg.width,cfg.height])
                loc_targets.append(loc_target)
                cls_targets.append(cls_target)
        if not self.test:
            return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
        return inputs
