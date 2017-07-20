import scipy.misc
from scipy import ndimage
import sys
import os
import glob
import json
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl




def data_import(directory_path):
            
    
    path = directory_path + "*"  
    filenames = glob.glob(path)

    processed_image_count = 0
    useful_image_count = 0

    X = []
    
    img_f = scipy.misc.imread(filenames[0])
    height, width, chan = img_f.shape
    

    for filename in filenames:
        processed_image_count += 1
        img = scipy.misc.imread(filename)
        assert chan == 3
        img = scipy.misc.imresize(img, size=(height, width), interp='bilinear')
        X.append(img)
        useful_image_count += 1
        
    print("processed %d, used %d" % (processed_image_count, useful_image_count))

    X = np.array(X).astype(np.float32)
    
    return X




def cell_crop(X, binimg_thred = 80., min_area=4000, scale_h=128, scale_v=128, chs=0):
    
    
    cells = np.empty((0, scale_v*2, scale_h*2, 3))
    
    for i in range(X.shape[0]):
        img = X[i].astype(np.uint8)
        img_chs = cv2.split(img)
        img_preprocessed = cv2.GaussianBlur(img_chs[chs],(5,5),0)
        binimg = (img_preprocessed > np.percentile(img_preprocessed, binimg_thred))
        binimg = binimg.astype(np.uint8)

        img_, contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        arr=[]
    
        start=np.empty((0,2))
        start=np.append(start,np.array([[0, 0]]),axis=0)
    
        for j in contours:
            if cv2.contourArea(j)<min_area:
                continue
            x_=0
            y_=0
            for k in j:
                x_ += k[0][0]
                y_ += k[0][1]
            arr.append([x_/len(j), y_/len(j)])
        arr = np.array(arr)
    
    
        for j in range(len(arr)):
    
            if (arr[j][1] < scale_v) or (arr[j][1] > img.shape[0]-scale_v) or (arr[j][0] < scale_h) or (arr[j][0] > img.shape[1]-scale_h):
                continue 
        
            top = int(arr[j][1])-scale_v
            bottom = int(arr[j][1])+scale_v
    
            left = int(arr[j][0])-scale_h
            right = int(arr[j][0])+scale_h
    
            if left < 0:
                left = 0
                right = scale_h*2
            if right > img.shape[1]:
                right = img.shape[1]
                left = img.shape[1]-scale_h*2
    
            if top < 0:
                top = 0
                bottom = scale_v*2
            if bottom > img.shape[0]:
                bottom = img.shape[0]
                top = img.shape[0]-scale_v*2      
                
            img_crop = np.array(img[top:bottom,left:right]).reshape(scale_v*2, scale_h*2, 3).astype(np.uint8)
            img_chs = cv2.split(img_crop)
            img_preprocessed = cv2.GaussianBlur(img_chs[chs],(5,5),0)
            binimg = (img_preprocessed > np.percentile(img_preprocessed, binimg_thred))
            binimg = binimg.astype(np.uint8)

            img_, contours, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            contourArea = []
            
            for j in contours:
                contourArea.append(cv2.contourArea(j))
            contourArea_sum = sum(contourArea)
            if contourArea_sum<min_area:
                continue
    
            cells = np.append(cells,np.array(img[top:bottom,left:right]).reshape(1,scale_v*2, scale_h*2, 3),axis=0)

    print("cropped_cell_count:", cells.shape[0])
    
    fig = plt.figure(figsize=(10, 10))

    for i in range(cells.shape[0]):
    
        ax_cell = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
        ax_cell.imshow(cells[i].reshape((scale_v*2, scale_h*2, 3)).astype(np.uint8))

    plt.show()
    
    return cells

