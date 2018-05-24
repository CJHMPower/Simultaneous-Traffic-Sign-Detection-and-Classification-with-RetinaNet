import os

from preprocessing.bbox import BoundingBox
from preprocessing.errors import UnsupportedExtensionError, UnsupportedFormatError
import json

class AnnotationDir:
    def __init__(self, anno_file, labels):
        self.anno_file = anno_file
        self.labels = labels
        self.ann_dict = self.build_annotations()
        

    def build_annotations(self):
        box_dict = {}
        annos = json.loads(open(self.anno_file).read())

        for id in annos['imgs']:
            boxes = []
            for sign_dict in annos['imgs'][id]['objects']:
                label = sign_dict['category']
                
                left = int(sign_dict['bbox']['xmin'])
                right = int(sign_dict['bbox']['xmax'])
                top = int(sign_dict['bbox']['ymin'])
                bottom = int(sign_dict['bbox']['ymax'])
                box = BoundingBox(left, top, right, bottom, 512,512,self.labels.index(label))
                boxes.append(box)
            if len(boxes) > 0:
               
                box_dict[annos['imgs'][id]['path'].split('/')[-1]] = boxes
        return box_dict


    def get_boxes(self, fn):
        
        return self.ann_dict[fn]
