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
    parser.add_option('-d', '--img_dir', dest='img_dir', default='./test/3.jpg', 
                      help='number of epochs')
    parser.add_option('-m', '--model_path', dest='model_path', default='./outputs-0.ckpt/0-81600',
                      type='string', help='batch size')
    parser.add_option('-p', '--precision', dest='precision', default=10,
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
    #print(ans)
    #print(len(ans))
    #print(stop)

    if args.focal_degree==0:
        for i in range(len(ans)):    
            array.append(ans[i])
            count+=1
            if count%n ==0:
                
                graph.append(array)
                array=[]
                
        f = open("./predictions/area.txt",'w')        
        for i in range(len(graph)):
            print(graph[i]) 
            for j in range(len(graph[i])):
                #graph[i][j] = str(graph[i][j]).encode('utf-8')
                f.write(str(graph[i][j]))
                f.write(' ')
            f.write('\n')
            
        for i in range(len(images_)):
            #print(images_[i])
            pic_name='./predictions/images/crack_'+str(i)+'_'+str(int(utils.confidence_(logits_dict[i][0])))+'.jpg'
            #  print(semantic_cnn.confidence(logits_dict[i][0]))
            cv2.imwrite(pic_name,images_[i])
        cv2.imwrite('./predictions/prediction.jpg',image)
        f.close()
        drawing=utils.draw()
        drawing2=utils.draw_(images_path='./predictions/images/',image_path=path,area_path='./predictions/area.txt',red=255,green=0,blue=0)

       
        
        print("successfully save!")
                    
                    
    else: 
        for i in range(len(ans)):    
            array.append(ans[i])
            count+=1
            if count%stop ==0:
                
                graph.append(array)
                array=[]
                
        f = open("./predictions/area.txt",'w')        
        for i in range(len(graph)):
            print(graph[i]) 
            for j in range(len(graph[i])):
                #graph[i][j] = str(graph[i][j]).encode('utf-8')
                f.write(str(graph[i][j]))
                f.write(' ')
            f.write('\n')
        
        for i in range(len(images_)):
            #print(images_[i])
            pic_name='./predictions/images/crack_'+str(i)+'_'+str(int(utils.confidence_(logits_dict[i][0])))+'.jpg'
            #  print(semantic_cnn.confidence(logits_dict[i][0]))
            cv2.imwrite(pic_name,images_[i])
        cv2.imwrite('./predictions/prediction.jpg',image)
        f.close()
        drawing=utils.draw()
        
        print("successfully save!")
        print('The functions is not finished!')