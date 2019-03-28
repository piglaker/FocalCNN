# -*- coding: utf-8 -*-
                      
"""
Created on Tue Mar 19 11:02:23 2019

@author: Xiaoyin
"""
import os
import shutil
import cv2
import focal_cnn
import utils2
import argparse
normal_size=224
def get_args():
    parser = argparse.ArgumentParser()
    #parser = OptionParser()
    parser.add_argument('--img_dir','-d' , metavar='INPUT', nargs='+', 
                        default='E:\\zxtdeeplearning\\crack\\matlab_proc\\single\\1.jpg', help='dir of img')
    parser.add_argument('--output_dir','-o', 
                      default='E:\\zxtdeeplearning\\crack\\matlab_proc\\single_output', help='output_dir of result') 
    parser.add_argument('--model_path','-m',  default='./outputs-0.ckpt/0-81600',
                      help='batch size')
    parser.add_argument('--precision','-p' , default=5,
                       help='precision')
    parser.add_argument('--normal_size','-s',  
                      default=224, help='normal_size')
    parser.add_argument('--focal_degree', '-f', 
                      default=0, help='focal_degree:2~20')
    parser.add_argument('--method','-t', 
                      default='quick', help='method:quick or deep')
    parser.add_argument('--review','-r',  
                      default=False,help='method:quick or deep')
    parser.add_argument('--photo','-i',  
                      default='raw', help='output of photo:gray,raw,gradient')
    
    parser.add_argument('--admin','-a',  
                      default=True, help='administer')
    parser.add_argument('--free','-fr' , 
                      default=False, help='free')
    parser.add_argument('--unet','-u',  
                      default=True, help='unet')
    
    #(options, args) = parser.parse_args()
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    n=args.precision
    path=args.img_dir
    
    img=cv2.imread(path)
    output_dir=args.output_dir
  
    ans,image,logits_dict,images_,stop=focal_cnn.detection(img,args.model_path,args.precision,args.normal_size,args.focal_degree,args.method,args.review,args.photo,args.admin)
    
    if os.path.exists(output_dir+'/images'):
        shutil.rmtree(output_dir+'/images')
    os.makedirs(output_dir+'/images')
    img=cv2.resize(img,(args.precision*normal_size,args.precision*normal_size),interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('E:\\zxtdeeplearning\\crack\\matlab_proc\\original_pic\\1.png',img)    
    
    #print(len(images_))
    #print(len(logits_dict))
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
        f = open(output_dir+"/area.txt",'w')        
        for i in range(len(graph)):
            print(graph[i]) 
            for j in range(len(graph[i])):
                #graph[i][j] = str(graph[i][j]).encode('utf-8')
                f.write(str(graph[i][j]))
                f.write(' ')
            f.write('\n')            
        for o in range(len(images_)): 
            pic_name=output_dir+'/images/crack_'+str(o)+'_.jpg'
            #  print(semantic_cnn.confidence(logits_dict[i][0]))
            cv2.imwrite(pic_name,images_[o])
        
        cv2.imwrite(output_dir+'/prediction.jpg',image)
        f.close()
        
        
        #cv2.imwrite(output_dir+'/prediction_t.jpg',utils2.enhance(utils2.gradient(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY))))
        #utils.unet()
        
        #drawing=utils2.draw()
        utils2.paint(output_dir)
        #drawing2=utils2.draw_(images_path=output_dir+'/images/',image_path=path,area_path=output_dir+'/area.txt',red=255,green=0,blue=0)    
        #utils2.clear(path=output_dir+'/images')
        print("successfully save!")        
    else:       
        for i in range(len(ans)):    
            array.append(ans[i])
            count+=1
            if count%stop ==0:                
                graph.append(array)
                array=[]                
        f = open(output_dir+'/area.txt','w')        
        for i in range(len(graph)):
            print(graph[i]) 
            for j in range(len(graph[i])):
                #graph[i][j] = str(graph[i][j]).encode('utf-8')
                f.write(str(graph[i][j]))
                f.write(' ')
            f.write('\n')        
        for i in range(len(images_)):
            #print(images_[i])
            pic_name=output_dir+'/images/crack_'+str(i)+'_.jpg'
            #  print(semantic_cnn.confidence(logits_dict[i][0]))
            cv2.imwrite(pic_name,images_[i])
        cv2.imwrite(output_dir+'/prediction.jpg',image)
        f.close()
        #drawing=utils2.draw()
        utils2.paint(output_dir)   
        #utils2.clear(path=output_dir+'/images')
        print("successfully save!")
        print('The functions is not finished!')