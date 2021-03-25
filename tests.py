import utils
import os
import random
import numpy as np

from torch.utils import data
from datasets.cityscapes import Cityscapes
from utils import ext_transforms as et

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt

data_root = '/data/kdn/Dataset/Cityscapes'
crop_size = 520
train_transform = et.ExtCompose([
            et.ExtRandomCrop(size=(crop_size, crop_size)),
            et.ExtColorJitter( brightness=0.5, contrast=0.5, saturation=0.5 ),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
        ])

val_transform = et.ExtCompose([
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

train_dst = Cityscapes(root=data_root,
                    split='train', transform=train_transform)
val_dst = Cityscapes(root=data_root,
                    split='val', transform=val_transform)

train_loader = data.DataLoader(
                    train_dst, batch_size=2, shuffle=True, num_workers=2)
val_loader = data.DataLoader(
                    val_dst, batch_size=2, shuffle=True, num_workers=2)

for i,data in enumerate(train_loader):
    images,depths,labels = data
    if i == 0:
        img = images[0].numpy()
        img = np.transpose(img,[1,2,0])
        print(img.shape)
        img = img[:,:,::-1]

        lbl = labels[0].numpy()
        print(lbl.shape)

        dep = depths[0].numpy()
        dep = np.transpose(dep,[1,2,0])
        print(dep.shape)
        dep = dep[:,:,::-1]

        
        plt.figure()
        plt.subplot(3,1,1)
        plt.imshow(img)
        plt.subplot(3,1,2)
        plt.imshow(lbl)
        plt.subplot(3,1,3)
        plt.imshow(dep)
        plt.show()