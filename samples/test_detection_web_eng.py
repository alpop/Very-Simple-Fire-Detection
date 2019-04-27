# coding: utf-8 python 3.5.2 Ubuntu 16.04
"""Created on Sat Jan 14 21:48:45 2017
Author: Alexander Popov. 2017-2019
GPL V2
"""

import sys

#Models, scripts and images are here
model_dir  = '../models/' 
sys.path.append('..')
images    = './images.url' 
 
import logging
import warnings
import pandas as pd

#Disable warnings
logging.getLogger('requests').setLevel(logging.CRITICAL)
warnings.simplefilter('ignore')

import mxnet as mx # mx.__version__  '0.11.0'
import datetime
import IPython.display as dp
from PIL import Image
import fire_detection as f
import urllib.request

#Get image urls
links = pd.read_csv(images,  header=None)
#Load  models

model_1K = f.model_1K(model_dir)
mean_img = mx.nd.load(model_dir + 'Inception/mean_224.nd')['mean_img']
model_21K = f.model_21K(model_dir)

#Prepare for test loop
max_images = 1000 # stop after processing this number of images
count     = 0     # image count
errcount  = 0     # count of images in error
start_ = datetime.datetime.today()
total_fires = 0
count = 0
  
#Test loop
for i in range(len(links)):  
    if count >= max_images:
        break        
    count += 1                
    try:                                                
         link = links[0][i]
         print (i+1,link)
         
         #Read image       
         img, hdr = urllib.request.urlretrieve(link)         
         image = Image.open(img)
         dp.display(image)

         #Detect fire
         Fire = f.Detection(image_file=img,
                            mean_img = mean_img, 
                            model_dir=model_dir,
                            model_1K=model_1K, 
                           model_21K=model_21K)
         #Print results
         if Fire:
             fire_class = 'Yes'
             total_fires += 1
         else:
             fire_class = 'No'   
         print ('Fire: ', fire_class, '\n')
    except OSError as err:
         print ('')
         print (err)
         errcount += 1
         pass
    except ValueError as err:
         print ('')
         print (err)
         errcount += 1
         pass
 
    if count >= max_images :
         break
                            
print ('Pictures processed : ', count)
print ('Unreadable: ', errcount)
print ('Fire detected: ',total_fires)
end_ = datetime.datetime.today()
print ('\nStart:  ', start_)
print ('End:    ',   end_)
elapsed = end_ - start_
print ('Elapsed:', elapsed)
print ('Speed (images per second):', round(count/elapsed.total_seconds(),2))
#End of test