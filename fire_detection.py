# coding: utf-8 python 3.5.2 Ubuntu 16.04
"""
Created on Sat Jan 14 21:48:45 2017
Fire detection
Author: Alexander Popov. 2017
GPL V2
"""

import mxnet as mx
import numpy as np
from skimage import io, transform
from collections import namedtuple
import cv2

#Load 1K model
def model_1K(model_dir): 
    # Load  pre-trained 1K model
    prefix = model_dir + 'Inception/Inception_BN'
    num_round = 39
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, num_round)
    #model_1K = mx.mod.Module(symbol=sym, context=mx.cpu())
    model_1K = mx.mod.Module(symbol=sym, context=mx.gpu())
    model_1K.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    model_1K.set_params(arg_params, aux_params)
    return model_1K
#Load 21K model    
def model_21K(model_dir):
    # Load pre-trained 21K model
    prefix = model_dir + 'Full ImageNet Network/Inception'
    num_round = 9
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, num_round)
    #model_21K = mx.mod.Module(symbol=sym, context=mx.cpu())
    model_21K = mx.mod.Module(symbol=sym, context=mx.gpu())
    model_21K.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    model_21K.set_params(arg_params, aux_params)
    return model_21K
   
#Preprocess image
def PreprocessImage(path,mean_img = None):
    # Read image
    img = io.imread(path)
    # We crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # Resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    normed_img = np.asarray(resized_img) * 256
    # Swap axes to make image from (224, 224, 4) to (3, 224, 224)
    normed_img = np.swapaxes(normed_img, 0, 2)
    normed_img = np.swapaxes(normed_img, 1, 2)  
    # Substract mean 
    if mean_img is not None:
        normed_img = normed_img - mean_img.asnumpy()
        
        if 0: #For debug purposes
            n = normed_img            
            n = np.swapaxes(normed_img, 0, 2)
            n = np.swapaxes(normed_img, 0, 1)
            cv2.namedWindow('Mxnet', cv2.WINDOW_NORMAL)
            cv2.imshow('Mxnet', n.astype(np.uint8).transpose((1,2,0))[:,:,[2,1,0]])
            cv2.waitKey(1)

    y = np.ascontiguousarray(normed_img)
    y.resize(1, 3, 224, 224)
    return y
   
#Detect fire
def Detection(image_file=None,mean_img = None, model_dir='',model_1K=None, model_21K=None):
    #Keyword definitions
    keywords  = ['fire','volcano','smoke','pyromaniac','flame', 'geyser','torch']
    keywords2 = ['wow!!!']
    #Load synsets (text labels)
    synset_1K = [l.strip()  for l in open(model_dir + 'Inception/synset.txt').readlines()]
    synset_21K = [l.strip() for l in open(model_dir + 'Full ImageNet Network/synset.txt').readlines()]  
    #Get preprocessed batch with mean substraction
    batch = PreprocessImage(image_file,mean_img = mean_img)      
    # Get prediction probability of 1K classes from model
    Batch = namedtuple('Batch', ['data'])
    model_1K.forward(Batch([mx.nd.array(batch)]))
    prob = model_1K.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)      
    #Argsort, get prediction index from largest prob to lowest
    pred = np.argsort(prob)[::-1]
    # Get top labels
    top = [synset_1K[pred[i]] for i in range(13)]
    # Detect fire in 1K classes
    fire_1K = False
    for i in range(len(top)):
        for j in range(len(keywords)):                       
           sub = keywords[j]                       
           if sub in top[i].lower() and  keywords2[0] not in top[i].lower():        
             fire_1K = True 
    # Get prediction probability of 21K classes from model
    batch = PreprocessImage(image_file,mean_img = mean_img)
    model_21K.forward(Batch([mx.nd.array(batch)]))
    prob = model_21K.get_outputs()[0].asnumpy()
    prob = np.squeeze(prob)    
    # Argsort, get prediction index from largest prob to lowest
    pred = np.argsort(prob)[::-1]
    # Get top labels
    top = [synset_21K[pred[i]] for i in range(5)]   
    # Detect fire in 1K classes
    fire_21K = False
    for i in range(len(top)):
        for j in range(len(keywords)):                       
           sub = keywords[j]                       
           if sub in top[i].lower() and  keywords2[0] not in top[i].lower():        
             fire_21K = True  
    if fire_21K or fire_1K:
         Fire = True
    else:
         Fire = False
    return Fire                             
#End of file