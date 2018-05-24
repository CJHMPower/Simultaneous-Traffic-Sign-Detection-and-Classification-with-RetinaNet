import numpy as np
import random
from PIL import Image

import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, example):
        for t in self.transforms:
            example = t(example)
        return example


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, example):
        image, boxes, labels = example['image'], example['boxes'], example['labels']
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'image': image, 'boxes': boxes, 'labels': labels}


class Scale:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, example):
        image, boxes = example['image'], example['boxes']
       
        width, height = image.size
        if isinstance(self.output_size, int):
            if width < height:
                new_width, new_height = width / height * self.output_size, self.output_size
            else:
                new_width, new_height = self.output_size, height / width * self.output_size
        else:
            new_width, new_height = self.output_size
        new_width, new_height = int(new_width), int(new_height)
        image=image.resize((new_width, new_height))
        boxes=[box.resize(new_width, new_height) for box in boxes]
        
        return {'image': image, 'boxes': boxes}


class RandomHorizontalFlip:
    def __call__(self, example):
        image, boxes = example['image'], example['boxes']
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = [box.flip() for box in boxes]
        return {'image': image, 'boxes': boxes}


class ToTensor:
    def __call__(self, example):
        image, boxes = example['image'], example['boxes']
        image = np.array(image).transpose((2, 0, 1))
        labels = np.array([box.label for box in boxes])
        boxes = np.array([[box.left, box.top, box.right, box.bottom] for box in boxes])
        image, boxes, labels = torch.from_numpy(image), torch.from_numpy(boxes), torch.from_numpy(labels)
        if isinstance(image, torch.ByteTensor):
            image = image.float().div(255)
        return {'image': image, 'boxes': boxes, 'labels': labels}


class Unnormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
