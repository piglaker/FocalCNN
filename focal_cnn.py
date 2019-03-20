# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:48:46 2019

@author: Xiaoyin
"""

import tensorflow as tf
import cv2
import utils
#normal_size=1296
#cv2.rectangle(img,(x,y),(x2,y2),(0,255,0),width)

def detection(image,model_path,n,normal_size=224,focal_degree=0,method='quick',review=False,photo='gradient'):
    '''   
    inputs:image
           model_path
           number of cut_pic
           normal_size        
            focal or not
            method
            review or not
            gradient or not
    outputs:area(express crack of the image)
            image:labeled image
            images_:gradient target
            logits_dict: logits of the target
    '''
    stop=0 
    area=[]   # matrix of 0 1
    predicted_label_dict=[]  # predictedlabel
    predicted_logits_dict=[] #predicted logits 
    images_1=[]#1 images(crack)
    images_0=[]#0 images(none)
    logits_dict=[]#logits
    launch=n*normal_size
    
    for i in range(n*n):
        area.append(0)   
        
    
    #judge=select_judge(method)    
    if focal_degree!=0:
        #自动对焦模式
        print('Focal starts!It will take longer time...')
        area,image,logits_dict,images_1,stop=focal(image,model_path,focal_degree,photo)        
    
        return area,image,logits_dict,images_1,stop
    
    else:       
        image = cv2.resize(image,(launch,launch),interpolation=cv2.INTER_CUBIC)
        images=utils.pic_cut(image,n,normal_size=224)
    
    if method !='quick':
        print('sorry,deep-method is still in fucking testing' )
  
    if method=='quick':               
            
        predicted_label_dict,predicted_logits_dict=cnn_predict(images,model_path)#predict
        #predict    
    
        packed_images=utils.pack_images_up(images,photo)
        '''     
        for i in range(len(packed_images)):                    
            if predicted_label_dict[i]==1:               
                area[i]=1;
                logits_dict.append(predicted_logits_dict[i])
                images_1.append(packed_images[i])
                y=normal_size*(int((i)/n))
                x=normal_size*((i)%n)
                cv2.rectangle(image,(x,y),(x+normal_size,y+normal_size),(0,255,0),3)
            else:
                images_0.append(images[i])
        '''
        area,image,logits_dict,images_1=utils.resort(area,packed_images,predicted_label_dict,predicted_logits_dict,image)
            
    else:
        '''
        deep:
        muilt-random-detection
        '''
        raw_image=0
        raw_image=image
        #cv2.imshow('0',raw_image)
        #cv2.waitKey(0)
        
        predicted_label_dict,predicted_logits_dict=cnn_predict(images,model_path)#predict
        #predict             
        
        packed_images=utils.pack_images_up(images,photo)
        
        
        for i in range(len(images)):                    
            if predicted_label_dict[i]==1:               
                area[i]=1
                logits_dict.append(predicted_logits_dict[i])
                images_1.append(packed_images[i])
                y=normal_size*(int((i)/n))
                x=normal_size*((i)%n)
                cv2.rectangle(image,(x,y),(x+normal_size,y+normal_size),(0,255,0),3)
            else:
                images_0.append(images[i])
                
        target,avg_confidence=utils.confidence(predicted_label_dict,predicted_logits_dict)
                
        print(target,avg_confidence)
                
        patience = int(input('Input the patience (0 ~ 10 interger):'))  
                    
                    
        with tf.Session() as sess:
            ckpt_path = model_path
            saver = tf.train.import_meta_graph(ckpt_path + '.meta')
            saver.restore(sess, ckpt_path)   
            inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')       
            classes = tf.get_default_graph().get_tensor_by_name('classes:0')
            logits = tf.get_default_graph().get_tensor_by_name('logits:0')#logits                            
            
        
            for i in range(0,patience+n,1):
                random_image,x,y=utils.random_cut_(raw_image,normal_size)  
                #cv2.imshow('0',random_image)
                
                label = sess.run(classes, feed_dict={inputs: [random_image]})
                logits_ = sess.run(logits, feed_dict={inputs: [random_image]})
                                
                print(utils.confidence_(logits_[0]))
                    
                if label==1:
                    if utils.confidence_(logits_[0])>avg_confidence:
                        cv2.rectangle(image,(x,y),(x+normal_size,y+normal_size),(255,255,0),3)
                        images_1.append(utils.pack_image_up_(random_image,photo))                             
                        logits_dict.append(logits_)
                                     
       
    if review:
            area,logits_dict,images_1_=review(images,model_path,area,logits_dict,images_1,images_0)
           

    return area,image,logits_dict,images_1,stop



def cnn_predict(images,model_path): 
    '''
    use trained-CNN to predict the cut_images 
    inputs:
        images
        model_path    
    outputs:
        label
        logits    
    '''
    predicted_label_dict=[]
    predicted_logits_dict=[]
    with tf.Session() as sess:
        ckpt_path = model_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)   
        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')       
        classes = tf.get_default_graph().get_tensor_by_name('classes:0')
        logits = tf.get_default_graph().get_tensor_by_name('logits:0')#logits                            
        for i in range(len(images)):
            image_batch=[images[i]]                                         
            predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
            predicted_label_dict.append(predicted_label)
            predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
            predicted_logits_dict.append(predicted_logits)          
    return  predicted_label_dict,predicted_logits_dict


def review(images,model_path,area,logits_dict,images_1,images_0):
    '''   
    waiting for ...
    '''
    area=area
    images_1_=images_1
    logits_dict=logits_dict

    return area,logits_dict,images_1_
    
'''
def judge(method,label,logits,images,images_):
      
    
    judge=0
    if method=='quick':
        judge=quick_judge(label)
    if method=='deep':
        judge=deep_judge(logits,images_)   
    return judge




def quick_judge(label):
    judge=0
    if label==1:
        judge=1
    return judge


    
def deep_judge(logits,images_,):
    judge=0
    #logits = tf.nn.softmax(logits)
    #confidence=confidence(logits)
    return judge

'''


def focal(image,model_path,focal_length,photo='raw'):
    '''
    focue_detection:
        auto for best cut search
    inputs:image
            model_path
            focal_length
    outputs:
            area,image,logits_dict,images_1,stop    
    preference:
        accuracy:more confidence
        aomount:more amount
    '''    
    raw_image=image
    max_target=0
    max_avg_score=0
    #loss_=0
    target=0
    avg_score=0
    score=0
    stop=0
    area=[]   # 
    predicted_label_dict=[]  #
    predicted_logits_dict=[] #
    logits_dict=[]
    images_1=[]
    normal_size=224    
    images_=[]
    predicted_label_dict_=[]
    predicted_logits_dict_=[]       
    model=input('Input the model you prefer (you hava to select quick or slow):')    
    #preference = input('Input the preference (you hava to select accuray or amount):')        
    with tf.Session() as sess:
        ckpt_path = model_path
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)   
        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')       
        classes = tf.get_default_graph().get_tensor_by_name('classes:0')
        logits = tf.get_default_graph().get_tensor_by_name('logits:0')#logits                                
        if model=='quick':    
            while True :    
                for i in range(2,focal_length,1):
                    target=0
                    score=0
                    avg_score=0
                    predicted_label_dict.clear()
                    predicted_logits_dict.clear()
                    launch=i*normal_size #  
                    image = cv2.resize(raw_image,(launch,launch),interpolation=cv2.INTER_CUBIC)
                    images=utils.pic_cut(image,i,normal_size=224)                               
                    #predicted_label_dict,predicted_logits_dict=cnn_predict(images,model_path)                        
                    ''' 
                    if len(images)<50:
                        for o in range(len(images)):
                            image_batch=[images[o]]                                         
                            predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                            predicted_label_dict.append(predicted_label)
                            predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                            predicted_logits_dict.append(predicted_logits)    
                    else:
                        o=0
                        t=len(images)
                        while t>50:
                            t=t-50
                            o+=50
                            image_batch=[images[o:o+50]]                                         
                            predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                            predicted_label_dict.append(predicted_label)
                            predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                            predicted_logits_dict.append(predicted_logits)
                        image_batch=[images[o:len(images)]]                                         
                        predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                        predicted_label_dict.append(predicted_label)
                        predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                        predicted_logits_dict.append(predicted_logits)
                     '''
                    for o in range(len(images)):
                        image_batch=[images[o]]                                         
                        predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                        predicted_label_dict.append(predicted_label)
                        predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                        predicted_logits_dict.append(predicted_logits)              
                    for j in range(len(predicted_label_dict)):
                        if predicted_label_dict[j]==1:
                            target+=1
                            #print(predicted_logits_dict[j])
                            score=score+utils.confidence_(predicted_logits_dict[j][0])
                            avg_score=score/target
                    print(target,avg_score,i)
                    if i==2 :
                        max_target=target
                        max_avg_score=avg_score
                        images_=images
                        predicted_label_dict_=predicted_label_dict
                        stop=2
                    else:             
                        images_=images
                        predicted_label_dict_=predicted_label_dict
                        predicted_logits_dict_=predicted_logits_dict
                        stop=i
                        if utils.gradient_function(max_target,max_avg_score,target,avg_score,preference='accuracy'):
                            break
                        else :
                            max_avg_score=avg_score
                            max_target=target                    
                break
        else:
            for i in range(2,focal_length,1):
                target=0
                score=0
                avg_score=0
                predicted_label_dict.clear()
                predicted_logits_dict.clear()
                launch=i*normal_size #  
                image = cv2.resize(raw_image,(launch,launch),interpolation=cv2.INTER_CUBIC)
                images=utils.pic_cut(image,i,normal_size=224)                          
                #predicted_label_dict,predicted_logits_dict=cnn_predict(images,model_path)                            
                '''
                if len(images)<50:
                    for o in range(len(images)):
                        image_batch=[images[o]]                                         
                        predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                        predicted_label_dict.append(predicted_label)
                        predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                        predicted_logits_dict.append(predicted_logits)    
                else:
                    o=0
                    t=len(images)
                    while t>50:
                        t=t-50
                        o+=50
                        image_batch=[images[o:o+50]]                                         
                        predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                        predicted_label_dict.append(predicted_label)
                        predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                        predicted_logits_dict.append(predicted_logits)
                    image_batch=[images[o:len(images)]]                                         
                    predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                    predicted_label_dict.append(predicted_label)
                    predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                    predicted_logits_dict.append(predicted_logits)                    
                '''
                for o in range(len(images)):
                        image_batch=[images[o]]                                         
                        predicted_label = sess.run(classes, feed_dict={inputs: image_batch})
                        predicted_label_dict.append(predicted_label)
                        predicted_logits = sess.run(logits, feed_dict={inputs: image_batch})
                        predicted_logits_dict.append(predicted_logits) 
                for j in range(len(predicted_label_dict)):
                    if predicted_label_dict[j]==1:
                        target+=1
                        #print(predicted_logits_dict[j])
                        score=score+utils.confidence_(predicted_logits_dict[j][0])
                        avg_score=score/target
                print(target,avg_score,i)
                if i==2 :   
                    max_target=target
                    max_avg_score=avg_score
                    images_=images
                    predicted_label_dict_=predicted_label_dict
                    stop=2
                else:             
                    images_=images
                    predicted_label_dict_=predicted_label_dict
                    predicted_logits_dict_=predicted_logits_dict
                    stop=i
                    if utils.max_function(max_target,max_avg_score,target,avg_score,preference='accuracy'):
                        max_avg_score=avg_score
                        max_target=target
       
    images_=utils.pack_images_up(images_,photo)          
    for i in range(stop*stop):
        area.append(0)    
    launch=stop*normal_size #  
    image = cv2.resize(raw_image,(launch,launch),interpolation=cv2.INTER_CUBIC)              
    for i in range(len(predicted_label_dict_)):
        if predicted_label_dict_[i]==1:
            area[i]=1;
            logits_dict.append(predicted_logits_dict_[i])
            images_1.append(images_[i])
            y=normal_size*(int((i)/stop))
            x=normal_size*((i)%stop)
            cv2.rectangle(image,(x,y),(x+normal_size,y+normal_size),(0,255,0),3)                    
    '''
    area,image,logits_dict,images_1=utils.resort(area,packed_images,predicted_label_dict,predicted_logits_dict,image)
    '''
    
    return area,image,logits_dict,images_1,stop

