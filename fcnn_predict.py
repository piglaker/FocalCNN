# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:02:23 2019

@author: Xiaoyin
"""

import cv2
import focal_cnn
import utils
from optparse import OptionParser

def get_args():
    parser = OptionParser()
    parser.add_option('-d', '--img_dir', dest='img_dir', default='./test/4.jpg', 
                      help='number of epochs')
    parser.add_option('-m', '--model_path', dest='model_path', default='./outputs-0.ckpt/0-81600',
                      type='string', help='batch size')
    parser.add_option('-p', '--precision', dest='precision', default=5,
                      type='int', help='precision')
    parser.add_option('-s', '--normal_size', dest='normal_size',
                      default=224, help='normal_size')
    parser.add_option('-f', '--focal_degree', dest='focal_degree',
                      default=0, help='focal_degree:2~20')
    parser.add_option('-t', '--method', dest='method',
                      default='quick', help='method:quick or deep')
    parser.add_option('-r', '--review', dest='review',
                      default=False,help='method:quick or deep')
    parser.add_option('-i', '--image', dest='photo',
                      default='raw', help='output of photo:gray,raw,gradient')
    parser.add_option('-o', '--output_dir', dest='output_dir',
                      default='./predictions', help='output_dir of result')
    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()
    n=args.precision
    
    path=args.img_dir
    img=cv2.imread(path)
    
    ans,image,logits_dict,images_,stop=focal_cnn.detection(img,args.model_path,args.precision,args.normal_size,args.focal_degree,args.method,args.review,args.photo)

    graph=[]
    array=[]
    count=0


    if args.focal_degree==0:
        for i in range(len(ans)):    
            array.append(ans[i])
            count+=1
            if count%n ==0:
                
                graph.append(array)
                array=[]
        for i in range(len(graph)):
            print(graph[i])
        for i in range(len(images_)):
            #print(images_[i])
            pic_name='./predictions/crack_'+str(i)+'_'+str(int(utils.confidence_(logits_dict[i][0])))+'.jpg'
            #  print(semantic_cnn.confidence(logits_dict[i][0]))
            cv2.imwrite(pic_name,images_[i])
        cv2.imwrite('./predictions/prediction.jpg',image)
        print("successfully save!")
                    
                    
    else: 
        print("successfully save!")
        for i in range(len(images_)):
            #print(images_[i])
            pic_name='./predictions/crack_'+str(i)+'_'+str(int(utils.confidence_(logits_dict[i][0])))+'.jpg'
            #  print(semantic_cnn.confidence(logits_dict[i][0]))
            cv2.imwrite(pic_name,images_[i])
        cv2.imwrite('./predictions/prediction.jpg',image)
        print('The functions not finished!')