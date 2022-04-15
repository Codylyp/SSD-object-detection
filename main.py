import argparse
import os
import numpy as np
import time
import cv2

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
import os
import matplotlib.pyplot as plt

from dataset import *
from model import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
#please google how to use argparse
#a short intro:
#to train: python main.py
#to test:  python main.py --test


class_num = 4 #cat dog person background

#num_epochs = 100
num_epochs = 100
batch_size = 32
#batch_size = 8


boxs_default = default_box_generator([10,5,3,1], [0.2,0.4,0.6,0.8], [0.1,0.3,0.5,0.7])


#Create network
network = SSD(class_num)
network.cuda()
cudnn.benchmark = True


if not args.test:
    #dataset = COCO("small/train/images/", "small/train/annotations/", class_num, boxs_default, train = True, test = False, crop = False, image_size=320)
    #dataset_test = COCO("small/train/images/", "small/train/annotations/", class_num, boxs_default, train = False, test = False, crop = False, image_size=320)
    dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, test = False, crop = True, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, test = False, crop = True, image_size=320)
    #dataset = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = True, test = False, crop = False, image_size=320)
    #dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, test = False, crop = False, image_size=320)
    
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = optim.Adam(network.parameters(), lr = 1e-4)
    #feel free to try other optimizers and parameters.
    
    start_time = time.time()

    #train_loss_sum = 0.0
    train_loss_total = []
    #validation_loss_sum = 0.0
    validation_loss_total = []
    #batches_per_epoch = len(train_loader.batch_sampler)
    for epoch in range(num_epochs):
        #TRAINING
        network.train()

        avg_loss = 0
        avg_count = 0
        #filename = []
        for i, data in enumerate(dataloader, 0):
            images_, ann_box_, ann_confidence_, f_name, height, width = data
            #print(f_name, "file name")
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            optimizer.zero_grad()
            pred_confidence, pred_box = network(images)
            #print("done network")
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            #print("done loss")
            loss_net.backward()
            #print("done loss backward")
            optimizer.step()
            
            avg_loss += loss_net.data
            avg_count += 1
            #filename.append(f_name)
            #write_file(pred_box.detach().cpu().numpy(), pred_confidence.detach().cpu().numpy(), boxs_default, f_name, height.detach().cpu().numpy(), width.detach().cpu().numpy())#####################

        print('[%d] time: %f train loss: %f' % (epoch, time.time()-start_time, avg_loss/avg_count))
        train_loss_total.append(avg_loss.detach().cpu().numpy()/avg_count)##########################

        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        #print(pred_box_.shape, "pred")
        #write_file(pred_box.detach().cpu().numpy(), pred_confidence.detach().cpu().numpy(), boxs_default, filename)#####################
        #print(images_[0].numpy())
        visualize_pred(("train" + f_name[0]), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        #print(pred_box_.shape, "before")
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        visualize_pred(("train" + f_name[0] + "nms"), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        
        
        #VALIDATION
        network.eval()
        
        # TODO: split the dataset into 90% training and 10% validation
        # use the training set to train and the validation set to evaluate
        vali_loss = 0
        vali_count = 0

        for i, data in enumerate(dataloader_test, 0):
            images_, ann_box_, ann_confidence_, f_name, height, width = data
            images = images_.cuda()
            ann_box = ann_box_.cuda()
            ann_confidence = ann_confidence_.cuda()

            pred_confidence, pred_box = network(images)
            #########################
            pred_confidence_ = pred_confidence.detach().cpu().numpy()
            pred_box_ = pred_box.detach().cpu().numpy()
            ##########################
            loss_net = SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box)
            #loss_net.backward()
            vali_loss += loss_net.data
            vali_count += 1
            ###########################
            #optional: implement a function to accumulate precision and recall to compute mAP or F1.
            #update_precision_recall(pred_confidence_, pred_box_, ann_confidence_.numpy(), ann_box_.numpy(), boxs_default,precision_,recall_,thres)
        
        validation_loss_total.append(vali_loss.detach().cpu().numpy()/vali_count)##########################
        #visualize
        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        #visualize_pred("val", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        visualize_pred(("val"+str(epoch)), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        #visualize_pred(("val"+str(epoch)+"nms"), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        #optional: compute F1
        #F1score = 2*precision*recall/np.maximum(precision+recall,1e-8)
        #print(F1score)
        
        #save weights
        if epoch%10==9:
            #save last network
            print('saving net...')
            torch.save(network.state_dict(), 'network.pth')

    
    plt.plot(train_loss_total, color = 'b', label='Training loss')
    plt.plot(validation_loss_total, color = 'r', label='Validation loss')
    plt.legend()
    plt.title("The trend of training loss and validation loss during 100 epochs")
    plt.xticks(np.arange(len(train_loss_total)))
    plt.show()
    print(train_loss_total)
    print(validation_loss_total)
    plt.savefig('loss.png')


else:
    #TEST
    #dataset_test = COCO("small/test/images/", "small/test/annotations/", class_num, boxs_default, train = False, test = True, crop = False, image_size=320)
    dataset_test = COCO("data/train/images/", "data/train/annotations/", class_num, boxs_default, train = False, test = True, crop = False, image_size=320)
    #dataset_test = COCO("data/test/images/", "data/test/annotations/", class_num, boxs_default, train = False, test = True, crop = False, image_size=320)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0)
    network.load_state_dict(torch.load('network.pth'))
    network.eval()
    
    for i, data in enumerate(dataloader_test, 0):
        images_, ann_box_, ann_confidence_, f_name, height, width = data
        images = images_.cuda()
        ann_box = ann_box_.cuda()
        ann_confidence = ann_confidence_.cuda()

        pred_confidence, pred_box = network(images)

        pred_confidence_ = pred_confidence[0].detach().cpu().numpy()
        pred_box_ = pred_box[0].detach().cpu().numpy()
        
        #pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        
        #TODO: save predicted bounding boxes and classes to a txt file.
        #you will need to submit those files for grading this assignment
        #visualize_pred(("test" + f_name[0]), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        pred_confidence_,pred_box_ = non_maximum_suppression(pred_confidence_,pred_box_,boxs_default)
        #visualize_pred(("test" + f_name[0] + "nms"), pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)

        #######################################################################
        path = "train_txt/" + f_name[0] + ".txt"
        threshold = 0.5
        width = width.numpy()
        height = height.numpy()
        
        px = pred_box_[:,0]
        py = pred_box_[:,1]
        pw = pred_box_[:,2]
        ph = pred_box_[:,3]

        dx = boxs_default[:,0]
        dy = boxs_default[:,1]
        dw = boxs_default[:,2]
        dh = boxs_default[:,3]

        gx = px * dw + dx
        gy = py * dh + dy
        gw = np.exp(pw) * dw
        gh = np.exp(ph) * dh

        x_min = (gx - gw / 2) * width
        y_min = (gy - gh / 2) * height
        w = gw * width
        h = gh * height

        content = []
        for row in range(len(pred_confidence_)):
            #class_id = np.argmax(pred_conf[i, :3])
            for col in range(3):
                if pred_confidence_[row,col] > threshold:
                    content.append([col, x_min[row], y_min[row], w[row], h[row]])
            #class_id = np.argmax(pred_confidence_[row, :3])
            #content.append([class_id, x_min[row], y_min[row], w[row], h[row]])
        if(len(content) > 0):
            format_ = '%d','%1.2f','%1.2f','%1.2f','%1.2f'
            np.savetxt(path, content, format_)
        else:
            np.savetxt(path, content)

        #visualize_pred("test", pred_confidence_, pred_box_, ann_confidence_[0].numpy(), ann_box_[0].numpy(), images_[0].numpy(), boxs_default)
        #cv2.waitKey(1000)

