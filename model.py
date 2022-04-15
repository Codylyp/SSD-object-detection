import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F



def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    
    pred_confidence = torch.reshape(pred_confidence,(-1,4))
    pred_box = torch.reshape(pred_box,(-1,4))
    ann_confidence = torch.reshape(ann_confidence,(-1,4))
    ann_box = torch.reshape(ann_box,(-1,4))

    object_idx = torch.where(ann_confidence[:, 3] == 0)
    empty_idx = torch.where(ann_confidence[:, 3] == 1)

    object_idx = object_idx[0]
    empty_idx = empty_idx[0]

    object_conf = pred_confidence[object_idx]
    empty_conf = pred_confidence[empty_idx]

    object_coord = torch.where(ann_confidence[object_idx] == 1)
    empty_coord = torch.where(ann_confidence[empty_idx] == 1)

    object_target = object_coord[1]
    empty_target = empty_coord[1]

    l_cls = F.cross_entropy(object_conf, object_target) + 3 * F.cross_entropy(empty_conf, empty_target)
    l_box = F.smooth_l1_loss(pred_box[object_idx], ann_box[object_idx])

    return l_box + l_cls


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()
        
        self.class_num = class_num #num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        #self.
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
            )

        self.left_block_1 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()            
            )

        self.left_block_2 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()            
            )

        self.left_block_3 = nn.Sequential(
            nn.Conv2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()            
            )

        self.left_box = nn.Conv2d(256, 16, 1, 1)
        self.left_conf = nn.Conv2d(256, 16, 1, 1)

        self.right_box_1 = nn.Conv2d(256, 16, 3, 1, padding = 1)
        self.right_conf_1 = nn.Conv2d(256, 16, 3, 1, padding = 1)
        self.right_box_2 = nn.Conv2d(256, 16, 3, 1, padding = 1)
        self.right_conf_2 = nn.Conv2d(256, 16, 3, 1, padding = 1)
        self.right_box_3 = nn.Conv2d(256, 16, 3, 1, padding = 1)
        self.right_conf_3 = nn.Conv2d(256, 16, 3, 1, padding = 1)

        self.sm = nn.Softmax(dim=2)
        
    def forward(self, x):
        #input:
        #x -- images, [batch_size, 3, 320, 320]
        
        x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.
        
        #TODO: define forward
        x_prev = self.pre_layers(x)
        x1 = self.left_block_1(x_prev)
        x2 = self.left_block_2(x1)
        x3 = self.left_block_3(x2)

        box_4 = self.left_box(x3)
        box_4 = torch.reshape(box_4,(-1, 16, 1))
        conf_4 = self.left_conf(x3)
        conf_4 = torch.reshape(conf_4,(-1, 16, 1))

        box_1 = self.right_box_1(x_prev)
        box_1 = torch.reshape(box_1,(-1, 16, 100))
        conf_1 = self.right_conf_1(x_prev)
        conf_1 = torch.reshape(conf_1,(-1, 16, 100))

        box_2 = self.right_box_2(x1)
        box_2 = torch.reshape(box_2,(-1, 16, 25))
        conf_2 = self.right_conf_2(x1)
        conf_2 = torch.reshape(conf_2,(-1, 16, 25))

        box_3 = self.right_box_3(x2)
        box_3 = torch.reshape(box_3,(-1, 16, 9))
        conf_3 = self.right_conf_3(x2)
        conf_3 = torch.reshape(conf_3,(-1, 16, 9))

        bboxes = torch.cat([box_1,box_2,box_3,box_4], dim=2)
        bboxes = bboxes.permute(0,2,1)
        bboxes = torch.reshape(bboxes,(-1,540,4))

        confidence = torch.cat([conf_1,conf_2,conf_3,conf_4], dim=2)
        confidence = confidence.permute(0,2,1)
        confidence = torch.reshape(confidence,(-1,540,4))
        confidence = self.sm(confidence)

        #should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        
        #sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        #confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        #bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        
        return confidence, bboxes
