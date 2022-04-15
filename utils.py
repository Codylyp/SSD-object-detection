import numpy as np
import cv2
from dataset import iou, recover
#from dataset import iou
import math

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
#use [blue green red] to represent different classes

def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    #input:
    #windowname      -- the name of the window to display the images
    #pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    #ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    #image_          -- the input image to the network
    #boxs_default    -- default bounding boxes, [num_of_boxes, 8]
    
    _, class_num = pred_confidence.shape
    #class_num = 4
    class_num = class_num-1
    #class_num = 3 now, because we do not need the last class (background)
    
    image = np.transpose(image_, (1,2,0)).astype(np.uint8)# h,w,c
    image1 = np.zeros(image.shape,np.uint8)
    image2 = np.zeros(image.shape,np.uint8)
    image3 = np.zeros(image.shape,np.uint8)
    image4 = np.zeros(image.shape,np.uint8)
    image1[:]=image[:]
    image2[:]=image[:]
    image3[:]=image[:]
    image4[:]=image[:]
    #print(image_)
    #image1: draw ground truth bounding boxes on image1
    #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    #image3: draw network-predicted bounding boxes on image3
    #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    
    
    #draw ground truth
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i,j]>0.5: #if the network/ground_truth has high confidence on cell[i] with class[j]
                #TODO:
                #image1: draw ground truth bounding boxes on image1
                #image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
                
                #you can use cv2.rectangle as follows:
                #start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                #end_point = (x2, y2) #bottom right corner
                #color = colors[j] #use red green blue to represent different classes
                #thickness = 2
                #cv2.rectangle(image?, start_point, end_point, color, thickness)
                gx, gy, gw, gh = recover(ann_box[i], boxs_default[i])
                start_point_1 = (int(gx*image.shape[1] - gw*image.shape[1]/2), int(gy*image.shape[0] - gh*image.shape[0]/2))
                end_point_1 = (int(gx*image.shape[1] + gw*image.shape[1]/2), int(gy*image.shape[0] + gh*image.shape[0]/2))

                start_point_2 = (int(boxs_default[i,4]*image.shape[1]), int(boxs_default[i,5]*image.shape[0]))
                end_point_2 = (int(boxs_default[i,6]*image.shape[1]), int(boxs_default[i,7]*image.shape[0]))
                
                
                color = colors[j]
                thickness = 2
                cv2.rectangle(image1, start_point_1, end_point_1, color, thickness)
                cv2.rectangle(image2, start_point_2, end_point_2, color, thickness)
    
    #pred
    for i in range(len(pred_confidence)):
        for j in range(class_num):
            if pred_confidence[i,j]>0.5:
                #TODO:
                #image3: draw network-predicted bounding boxes on image3
                #image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                gx, gy, gw, gh = recover(pred_box[i], boxs_default[i])
                start_point_3 = (int(gx*image.shape[1] - gw*image.shape[1]/2), int(gy*image.shape[0] - gh*image.shape[0]/2))
                end_point_3 = (int(gx*image.shape[1] + gw*image.shape[1]/2), int(gy*image.shape[0] + gh*image.shape[0]/2))

                start_point_4 = (int(boxs_default[i,4]*image.shape[1]), int(boxs_default[i,5]*image.shape[0]))
                end_point_4 = (int(boxs_default[i,6]*image.shape[1]), int(boxs_default[i,7]*image.shape[0]))
                
                color = colors[j]
                thickness = 2
                cv2.rectangle(image3, start_point_3, end_point_3, color, thickness)
                cv2.rectangle(image4, start_point_4, end_point_4, color, thickness)

    #combine four images into one
    h,w,_ = image1.shape
    image = np.zeros([h*2,w*2,3], np.uint8)
    image[:h,:w] = image1
    image[:h,w:] = image2
    image[h:,:w] = image3
    image[h:,w:] = image4
    #print(image1)
    #cv2.imshow(windowname+" [[gt_box,gt_dft],[pd_box,pd_dft]]",image)
    #cv2.imwrite("images/" + windowname + ".jpg", image)
    cv2.imwrite("test_images2/" + windowname + ".jpg", image)
    cv2.waitKey(1)
    #if you are using a server, you may not be able to display the image.
    #in that case, please save the image using cv2.imwrite and check the saved image for visualization.

def cal_iou(boxs_pred, x_min,y_min,x_max,y_max):
    #input:
    #boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    #x_min,y_min,x_max,y_max -- another box (box_r)
    
    #output:
    #ious between the "boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]
    x4 = boxs_pred[:,0] - boxs_pred[:,2]/2
    x5 = boxs_pred[:,1] - boxs_pred[:,3]/2
    x6 = boxs_pred[:,0] + boxs_pred[:,2]/2
    x7 = boxs_pred[:,1] + boxs_pred[:,3]/2

    inter = np.maximum(np.minimum(x6,x_max)-np.maximum(x4,x_min),0)*np.maximum(np.minimum(x7,y_max)-np.maximum(x5,y_min),0)
    area_a = (x6-x4)*(x7-x5)
    area_b = (x_max-x_min)*(y_max-y_min)
    union = area_a + area_b - inter
    return inter/np.maximum(union,1e-8)


def non_maximum_suppression(confidence_, box_, boxs_default, overlap=0.5, threshold=0.5):
    #input:
    #confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    #box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    #boxs_default -- default bounding boxes, [num_of_boxes, 8]
    #overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    #threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.
    
    #output:
    #depends on your implementation.
    #if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    #you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    
    
    #TODO: non maximum suppression
    ###############################################
    confidence_dummy = np.delete(confidence_, -1, axis=1)
    max_list = [] # store the max value of each row
    #result_box = []
    #result_conf = []
    result_box = np.zeros((box_.shape[0], box_.shape[1]))
    result_conf = np.zeros((confidence_.shape[0], confidence_.shape[1]))

    #new_box_ = np.concate()
    #######################################
    xcenter_col = boxs_default[:,2]*box_[:, 0] + boxs_default[:,0]
    ycenter_col = boxs_default[:,3]*box_[:, 1] + boxs_default[:,1]
    w_col = boxs_default[:,2]*np.exp(box_[:,2])
    h_col = boxs_default[:,3]*np.exp(box_[:,3])
    xmin_col = xcenter_col - w_col/2
    ymin_col = ycenter_col - h_col/2
    xmax_col = xcenter_col + w_col/2
    ymax_col = ycenter_col + h_col/2
        
    xcenter_col = xcenter_col.reshape(len(xcenter_col), 1)
    ycenter_col = ycenter_col.reshape(len(ycenter_col), 1)
    w_col = w_col.reshape(len(w_col), 1)
    h_col = h_col.reshape(len(h_col), 1)
    xmin_col = xmin_col.reshape(len(xmin_col), 1)
    ymin_col = ymin_col.reshape(len(ymin_col), 1)
    xmax_col = xmax_col.reshape(len(xmax_col), 1)
    ymax_col = ymax_col.reshape(len(ymax_col), 1)

        #new_box_ = np.concatenate((box_, xmin_col, ymin_col, xmax_col, ymax_col), axis=1)
    new_box_ = np.concatenate((xcenter_col, ycenter_col, w_col, h_col, xmin_col, ymin_col, xmax_col, ymax_col), axis=1)
    pred_box_ = np.concatenate((xcenter_col, ycenter_col, w_col, h_col), axis=1)
    ####################################

    for i in range(confidence_.shape[0]):
        max_list.append(max(confidence_dummy[i]))

    max_list = np.array(max_list)
    highest = max(max_list)
    row_idx = np.argmax(max_list)

    while(highest > threshold):
        # move x from A to B
        x = box_[row_idx,:]
        
        result_box[row_idx,:] = x
        result_conf[row_idx,:] = confidence_[row_idx,:]
        
        box_[row_idx, :] = 0
        confidence_[row_idx, :] = 0
        #new_box_[i, :] = 0

        gx = pred_box_[:,0]
        gy = pred_box_[:,1]
        gw = pred_box_[:,2]
        gh = pred_box_[:,3]
        
        max_list[row_idx] = -1

        x_min = gx - gw/2
        y_min = gy - gh/2
        x_max = gx + gw/2
        y_max = gy + gh/2


        #ious = cal_iou(box_, x_min, y_min, x_max, y_max)
        ious = iou(new_box_, x_min, y_min, x_max, y_max)
        ious_true = ious > overlap

        for i in range(len(ious_true)):
            if ious_true[i] == True:
                #box_ = np.delete(box_, i, axis=0)
                #confidence_ = np.delete(confidence_, i, axis=0)
                #max_list = np.delete(max_list, i, axis=0)
                #boxs_default = np.delete(boxs_default, i, axis=0)#######################
                box_[i, :] = 0
                confidence_[i, :] = 0
                #new_box_[i, :] = 0
                max_list[i] = -1
        if len(max_list) != 0:
            highest = max(max_list)
            row_idx = np.argmax(max_list)
        else:
            #print("list is empty, in trouble")
            break

    return result_conf, result_box 

    ##############################################
    















