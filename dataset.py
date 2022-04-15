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
import numpy as np
import os
import cv2
import math
from PIL import Image
import random

#generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    #input:
    #layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    #large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    #small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].
    
    #output:
    #boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.
    
    #TODO:
    #create an numpy array "boxes" to store default bounding boxes
    #you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    #the first dimension means number of cells, 10*10+5*5+3*3+1*1
    #the second dimension 4 means each cell has 4 default bounding boxes.
    #their sizes are [ssize,ssize], [lsize,lsize], [lsize*sqrt(2),lsize/sqrt(2)], [lsize/sqrt(2),lsize*sqrt(2)],
    #where ssize is the corresponding size in "small_scale" and lsize is the corresponding size in "large_scale".
    #for a cell in layer[i], you should use ssize=small_scale[i] and lsize=large_scale[i].
    #the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    box_num = layers[0] * layers[0] + layers[1] * layers[1] + layers[2] * layers[2] + layers[3] * layers[3]
    box_num_total = len(layers) * (layers[0] * layers[0] + layers[1] * layers[1] + layers[2] * layers[2] + layers[3] * layers[3])
    boxes = np.empty([box_num, 4, 8])

    # y first, then x
    idx = 0
    for layer in range(len(layers)):
    	for x in range(layers[layer]):
    		for y in range(layers[layer]):

    			half_cell = 1/(layers[layer]*2)
    			x_center = round((x*(1/layers[layer]) + half_cell),2)
    			y_center = round((y*(1/layers[layer]) + half_cell),2)

    			box_size_1 = small_scale[layer]
    			box_size_2 = large_scale[layer]
    			box_width_3 = round((large_scale[layer]*math.sqrt(2)),2)
    			box_height_3 = round((large_scale[layer]/math.sqrt(2)),2)
    			box_width_4 = round((large_scale[layer]/math.sqrt(2)),2)
    			box_height_4 = round((large_scale[layer]*math.sqrt(2)),2)

    			x_min_1 = round((max(0, x_center - (box_size_1/2))),2)
    			x_max_1 = round((min(1, x_center + (box_size_1/2))),2)
    			y_min_1 = round((max(0, y_center - (box_size_1/2))),2)
    			y_max_1 = round((min(1, y_center + (box_size_1/2))),2)

    			x_min_2 = round((max(0, x_center - (box_size_2/2))),2)
    			x_max_2 = round((min(1, x_center + (box_size_2/2))),2)
    			y_min_2 = round((max(0, y_center - (box_size_2/2))),2)
    			y_max_2 = round((min(1, y_center + (box_size_2/2))),2)

    			x_min_3 = round((max(0, x_center - (box_width_3/2))),2)
    			x_max_3 = round((min(1, x_center + (box_width_3/2))),2)
    			y_min_3 = round((max(0, y_center - (box_height_3/2))),2)
    			y_max_3 = round((min(1, y_center + (box_height_3/2))),2)

    			x_min_4 = round((max(0, x_center - (box_width_4/2))),2)
    			x_max_4 = round((min(1, x_center + (box_width_4/2))),2)
    			y_min_4 = round((max(0, y_center - (box_height_4/2))),2)
    			y_max_4 = round((min(1, y_center + (box_height_4/2))),2)

    			boxes[idx, 0, :] = [x_center, y_center, box_size_1, box_size_1, x_min_1, y_min_1, x_max_1, y_max_1]
    			boxes[idx, 1, :] = [x_center, y_center, box_size_2, box_size_2, x_min_2, y_min_2, x_max_2, y_max_2] 
    			boxes[idx, 2, :] = [x_center, y_center, box_width_3, box_height_3, x_min_3, y_min_3, x_max_3, y_max_3] 
    			boxes[idx, 3, :] = [x_center, y_center, box_width_4, box_height_4, x_min_4, y_min_4, x_max_4, y_max_4]
    			idx = idx + 1 

    print(idx, "box_num should be 135")
    boxes = boxes.reshape([box_num_total, 8])
    boxes = np.array(boxes)
    return boxes


#this is an example implementation of IOU.
#It is different from the one used in YOLO, please pay attention.
#you can define your own iou function if you are not used to the inputs of this one.
def iou(boxs_default, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    
    inter = np.maximum(np.minimum(boxs_default[:,6],x_max)-np.maximum(boxs_default[:,4],x_min),0)*np.maximum(np.minimum(boxs_default[:,7],y_max)-np.maximum(boxs_default[:,5],y_min),0)
    area_a = (boxs_default[:,6]-boxs_default[:,4])*(boxs_default[:,7]-boxs_default[:,5])
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)



def match(ann_box,ann_confidence,boxs_default,threshold,cat_id,x_min,y_min,x_max,y_max):
    #input:
    #ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    #ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    #boxs_default            -- [num_of_boxes,8], default bounding boxes
    #threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    #cat_id                  -- class id, 0-cat, 1-dog, 2-person
    #x_min,y_min,x_max,y_max -- bounding box
    #ann_box = ann_box
    #ann_confidence = ann_confidence
    #compute iou between the default bounding boxes and the ground truth bounding box
    ious = iou(boxs_default, x_min,y_min,x_max,y_max)
    
    ious_true = ious>threshold
    #TODO:
    #update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    #if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    #this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    at_least_one = False

    gx = (x_min + x_max)/2
    gy = (y_min + y_max)/2
    gw = x_max - x_min
    gh = y_max - y_min

    for i in range(len(ious_true)):
        if ious_true[i] == True:
            at_least_one = True
            px = boxs_default[i][0]
            py = boxs_default[i][1]
            pw = boxs_default[i][2]
            ph = boxs_default[i][3]

            tx = (gx - px)/pw
            ty = (gy - py)/ph
            tw = math.log(gw/pw)
            th = math.log(gh/ph)

            ann_confidence[i, cat_id] = 1
            #ann_confidence[i, -1] = 0
            ann_confidence[i, 3] = 0
            ann_box[i, :] = [tx, ty, tw, th]

    
    ious_true = np.argmax(ious)
    #TODO:
    #make sure at least one default bounding box is used
    #update ann_box and ann_confidence (do the same thing as above)
    if at_least_one == False:
        px = boxs_default[ious_true][0]
        py = boxs_default[ious_true][1]
        pw = boxs_default[ious_true][2]
        ph = boxs_default[ious_true][3]

        tx = (gx - px)/pw
        ty = (gy - py)/ph
        tw = math.log(gw/pw)
        th = math.log(gh/ph)

        ann_confidence[ious_true, cat_id] = 1
        #ann_confidence[ious_true, -1] = 0
        ann_confidence[ious_true, 3] = 0
        ann_box[ious_true, :] = [tx, ty, tw, th]


    return ann_box, ann_confidence


def recover(ann_box,boxs_default):
    px = boxs_default[0]
    py = boxs_default[1]
    pw = boxs_default[2]
    ph = boxs_default[3]

    dx = ann_box[0]
    dy = ann_box[1]
    dw = ann_box[2]
    dh = ann_box[3]

    gx = pw*dx + px
    gy = ph*dy + py
    gw = pw*(math.exp(dw))
    gh = ph*(math.exp(dh))

    return gx, gy, gw, gh



class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, boxs_default, train = True, test = False, crop = True, image_size=320):
        self.train = train
        self.test = test################################
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        #overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.boxs_default = boxs_default
        self.box_num = len(self.boxs_default)
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        self.crop = crop
        
        #notice:
        #you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        if self.train == True and test == False:
        	self.img_names = self.img_names[:int(len(self.img_names)*0.9)]
        	#print(np.array(self.img_names).shape)
        elif self.train == False and test == False:
        	self.img_names = self.img_names[int(len(self.img_names)*0.9):]
        	#print(np.array(self.img_names).shape)

        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        #print(index, "index")
        ann_box = np.zeros([self.box_num,4], np.float32) #bounding boxes
        ann_confidence = np.zeros([self.box_num,self.class_num], np.float32) #one-hot vectors
        #one-hot vectors with four classes
        #[1,0,0,0] -> cat
        #[0,1,0,0] -> dog
        #[0,0,1,0] -> person
        #[0,0,0,1] -> background
        
        ann_confidence[:,-1] = 1 #the default class for all cells is set to "background"
        
        img_name = self.imgdir+self.img_names[index]
        #print(img_name)
        if self.test == False:
        	ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        	#print(self.img_names[index][:-3])
        	#print(self.img_names[index][:-4])
        	#print(ann_name)
        
        #TODO:
        #1. prepare the image [3,320,320], by reading image "img_name" first.
        #2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        #3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        #4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        ########################################
        image = cv2.imread(img_name)
        h = image.shape[0]
        w = image.shape[1]

       
        if self.crop == True and self.test == False:
            crop_xmin = 9999
            crop_ymin = 9999
            crop_xmax = -1
            crop_ymax = -1
            f = open(ann_name)
            data = f.readline()
            while data:
                data = data.strip()
                data = data.split(" ")
                #print(data)
                #class_id = int(data[0])
                crop_xmin = min(float(data[1]), crop_xmin)
                crop_ymin = min(float(data[2]), crop_ymin)
                curr_xmax = float(data[1]) + float(data[3])
                curr_ymax = float(data[2]) + float(data[4])
                crop_xmax = max(curr_xmax, crop_xmax)
                crop_ymax = max(curr_ymax, crop_ymax)
                data = f.readline()
            f.close()
            # do the random crop
            crop_xmin = int(random.uniform(0, crop_xmin))
            crop_ymin = int(random.uniform(0, crop_ymin))
            crop_xmax = int(random.uniform(crop_xmax, w))
            crop_ymax = int(random.uniform(crop_ymax, h))

            image_new = image[:, crop_xmin:, :]
            #image_new = image_new[:, :(crop_xmax-crop_xmin), :]
            image_new = image_new[crop_ymin:, :, :]
            #image_new = image_new[:(crop_ymax-crop_ymin), :, :]
            h = image_new.shape[0]
            w = image_new.shape[1]
            image = image_new

            # update ann
            f = open(ann_name)
            data = f.readline()
            while data:
                data = data.strip()
                data = data.split(" ")
                class_id = int(data[0])
                x_min = round(((float(data[1])-crop_xmin)/w),2)
                y_min = round(((float(data[2])-crop_ymin)/h),2)
                x_max = round(((float(data[1]) + float(data[3]) - crop_xmin)/w),2)
                y_max = round(((float(data[2]) + float(data[4]) - crop_ymin)/h),2)
                ann_box, ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
                #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
                data = f.readline()
            f.close()


        if self.crop == False and self.test == False:
        
            f = open(ann_name)
            data = f.readline()
            while data:
                #print(data)
                data = data.strip()
                data = data.split(" ")
                #print(data)
                class_id = int(data[0])
                
                x_min = float(data[1])/w
                #print(x_min)
                y_min = float(data[2])/h
                #print(y_min)
                x_max = x_min + float(data[3])/w
                y_max = y_min + float(data[4])/h
                
                ann_box, ann_confidence = match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
                #match(ann_box,ann_confidence,self.boxs_default,self.threshold,class_id,x_min,y_min,x_max,y_max)
                data = f.readline()
            f.close()

        
        image = cv2.resize(image, (self.image_size,self.image_size))
        image = np.transpose(image,(2,0,1))
        image = torch.from_numpy(image)

        return image, ann_box, ann_confidence, self.img_names[index][:-4], h, w

