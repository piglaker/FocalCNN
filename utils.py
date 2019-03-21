# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 21:55:15 2019

@author: Xiaoyin
"""

import numpy as np
import random
import math
import cv2
import os
import glob


def confidence_(logits):
    confidence=0
    confidence=-1*math.log(min(softmax(logits)))
    return confidence 


def confidence(predicted_label_dict,predicted_logits_dict):
    total_confidence=0
    target=0
    avg_confidence=0
    for i in range(len(predicted_label_dict)):
        if predicted_label_dict[i]==1:
            target+=1
            #print(predicted_logits_dict[i][0])
            total_confidence+=confidence_(predicted_logits_dict[i][0])
    avg_confidence=total_confidence/target    
    return target,avg_confidence


def softmax(array):
    '''
    softmax:regression
    '''
    s=0
    array2=[]
    for i in range(len(array)):
        s+=math.exp(array[i])
    for j in range(len(array)):
        array2.append(math.exp(array[j])/s)
    return array2



def make(a,b,image,normal_size=224):
    '''    
    create a img copy from image form (a,b)
    inputs: a,b, image, the size of copy
    outputs: a copy of image from (a,b) in size    
    '''
    img=tensor(normal_size,normal_size,3)    
    for i in range(normal_size):
        for j in range(normal_size):
            for u in range(3):
                img[i][j][u]=int(image[a+i][b+j][u])    
    return img



def tensor(a,b,c):
    '''
    return a tensor of 3-D(a,b,c)    
    '''
    img=np.zeros((b,c))
    image=[]
    for i in range(a):
        image.append(img)
    image=np.asarray(image)
    return image
# creat a tensor (a,b,c)
      
    

def pic_cut(image,n,normal_size=224):
    '''
    cut image into n     
    '''
    a,b=0,0
    launch=n*normal_size
    cuted_img=[]    
    image = cv2.resize(image,(launch,launch),interpolation=cv2.INTER_CUBIC)
    for i in range(n):    
        for j in range(n):    
            img=make(a,b,image,normal_size)
            cuted_img.append(img)
            b=b+normal_size
        a=a+normal_size
        b=0       
    return cuted_img


def random_cut_(image,normal_size=224):
    '''
    return a images from image (random)
    '''
    r1=int(random.random()*(len(image)-normal_size))
    r2=int(random.random()*(len(image[0])-normal_size))
    img=make(r1,r2,image,normal_size)
    return img,r1,r2


def random_cut(image,n,normal_size=224):
    '''
    get a random images of image
    '''
    images=[]    
    for i in range(n):
        r1=int(random.random()*(len(image)-normal_size))
        r2=int(random.random()*(len(image[0])-normal_size))
        img=make(r1,r2,image,normal_size)
        images.append(img)
    return images
    

def enhance(gray_image):
    image=gray_image
    for i in range(len(image)):
        for j  in range(len(image[i])):
            image[i][j]=int(math.pow(image[i][j],2)/255)

    return image
       
        
def gradient(image): 
    '''
    gradient   
    '''
    row, column = image.shape
    moon_f = np.copy(image)
    moon_f = moon_f.astype("float")
    gradient = np.zeros((row, column))
    for x in range(row - 1):
        for y in range(column - 1):
            gx = abs(moon_f[x + 1, y] - moon_f[x, y])
            gy = abs(moon_f[x, y + 1] - moon_f[x, y])
            gradient[x, y] = gx + gy            
    gradient = gradient.astype("uint8")
    return gradient


def pack_images_up(images,photo):
    '''
    pack images up into a list of images or gradient images
    '''    
    packed_images=[]
    if photo=='gradient':
        for i in range(len(images)):
            img=np.array(images[i])
            img=img.astype(np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)#注意：灰度图要求输入为np.ndarray dtype=uint8
            packed_images.append(gradient(cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)))#梯度图，注意梯度要去inputs为灰度图，
    #gray -resize
    if photo=='gray':
        for i in range(len(images)):
            img=np.array(images[i])
            img=img.astype(np.uint8)
            img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)#注意：灰度图要求输入为np.ndarray dtype=uint8
            packed_images.append(cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC))#      
    else:
        for i in range(len(images)):
            img=np.array(images[i])
            img=img.astype(np.uint8)
            packed_images.append(cv2.resize(img,(224,224)))
    return packed_images        


def pack_image_up_(image,photo):
    '''
    pack images up into a list of images or gradient images
    '''    
    packed_image=0
    if photo=='gradient':       
        img=np.array(image)
        img=img.astype(np.uint8)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)#注意：灰度图要求输入为np.ndarray dtype=uint8
        packed_image=gradient(cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC))#梯度图，注意梯度要去inputs为灰度图，
    #gray -resize
    if photo=='gray':
        img=np.array(image)
        img=img.astype(np.uint8)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)#注意：灰度图要求输入为np.ndarray dtype=uint8
        packed_image=cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)    
    else:
        img=np.array(image)
        packed_image=img.astype(np.uint8)
    return packed_image      

def resort(area,packed_images,predicted_label_dict,predicted_logits_dict,image,normal_size=224):
    logits_dict=[]
    images_1=[]
    images_0=[]
    n=int(math.sqrt(len(area)))
    print(n)
    for i in range(len(packed_images)):                    
            if predicted_label_dict[i]==1:               
                area[i]=1;
                logits_dict.append(predicted_logits_dict[i])
                images_1.append(packed_images[i])
                y=normal_size*(int((i)/n))
                x=normal_size*((i)%n)
                print('draw')
                cv2.rectangle(image,(x,y),(x+normal_size,y+normal_size),(0,255,0),3)
            else:
                images_0.append(packed_images[i])

    return area,image,logits_dict,images_1         

             
def gradient_function(max_target,max_avg_score,target,avg_score,preference):
    if preference=='amount':
        if max_target*0.7+max_avg_score*0.3>target*0.7+max_avg_score*0.3:
            return True
    if preference=='accuracy':
        if max_target*0.4+max_avg_score*0.7>target*0.4+max_avg_score*0.7:
            return True
    return False

            
def max_function(max_target,max_avg_score,target,avg_score,preference):
    if preference=='amount':
        if max_target*0.7+max_avg_score*0.3<target*0.7+max_avg_score*0.3:
            return True
    if preference=='accuracy':
        if max_target*0.4+max_avg_score*0.7<target*0.4+max_avg_score*0.7:
            return True
    return False            
 

           
def get_batch(images,a,b):
    image_batch = images[a:b+1]
    return image_batch
    
 
def draw_map(a,b,image,images,red=255,green=0,blue=0):
    '''
    draw the image from (a ,b) to ...  with color (red,green,blue)
    '''
    normal_size=224
    for i in range(a,a+normal_size,1):
        for j in range(b,b+normal_size,1):
            #print(a,b,i,j)
            if images[i-a][j-b]>=20:
                #print(i,j)
                image[i][j][0]=blue
                image[i][j][1]=green
                image[i][j][2]=red        
    return image





def draw(images_path='./predictions/images/',image_path='./predictions/prediction.jpg',area_path='./predictions/area.txt',red=255,green=0,blue=0):
    '''    
    draw the image according to the images,area    
    '''
    normal_size=224
    area=np.loadtxt(area_path)
    images_=[]
    images_path = os.path.join(images_path, '*.jpg')
    name=[]
    for image_file in glob.glob(images_path):
        img = cv2.imread(image_file,0)
        #print(image_file)
        #img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print(img)   
        img=gradient(img)
        images_.append(img)      
        label = int(image_file.split('_')[-2].split('_')[-1])
        name.append(label)  
    images=[]
    for i in range(len(name)):
        for j in range(len(name)):
            if name[j] ==i:
                images.append(images_[j])
    #sort
    image=cv2.imread(image_path)
    count=0
    for i in range(len(area)):
        for j in range(len(area[i])):
            if area[i][j]==1:
                #print(i,j)
                #print(len(images),count)
                image=draw_map(i*normal_size,j*normal_size,image,images[count],red,green,blue)     
                count+=1        
    cv2.imwrite('./predictions/draw_image.jpg',image)


def draw_(images_path,image_path,area_path='./predictions/area.txt',red=255,green=0,blue=0):
    '''    
    draw the image according to the images,area    
    '''
    normal_size=224
    area=np.loadtxt(area_path)
    images_=[]
    images_path = os.path.join(images_path, '*.jpg')
    name=[]
    for image_file in glob.glob(images_path):
        img = cv2.imread(image_file,0)
        #print(image_file)
        #img = cv2.resize(img,(224,224),interpolation=cv2.INTER_CUBIC)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #print(img)   
        img=gradient(img)
        images_.append(img)      
        label = int(image_file.split('_')[-2].split('_')[-1])
        name.append(label)  
    images=[]
    for i in range(len(name)):
        for j in range(len(name)):
            if name[j] ==i:
                images.append(images_[j])
    #sort
    image=cv2.imread(image_path)
    image = cv2.resize(image,(len(area)*normal_size,len(area)*normal_size),interpolation=cv2.INTER_CUBIC)
    
    count=0
    for i in range(len(area)):
        for j in range(len(area[i])):
            if area[i][j]==1:
                #print(i,j)
                #print(len(images),count)
                image=draw_map(i*normal_size,j*normal_size,image,images[count],red,green,blue)     
                count+=1        
    cv2.imwrite('./predictions/draw_image.jpg',image)

    
    
    
    