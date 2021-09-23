#!/usr/bin/env python
# coding: utf-8

# In[54]:


import tensorflow as tf
import numpy as np
import os


# In[100]:

#from tensorflow.python.client import device_lib
#def get_available_devices():
#    local_device_protos = device_lib.list_local_devices()
#    return [x.name for x in local_device_protos]
#print(get_available_devices()) 
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[465]:


# Training - All classes
# [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 29]
# Validation - All classes
# [2, 12, 30, 21, 14, 27]

# eval_dir = input("Where is the image stream: ")
# eval_dir = eval_dir.replace('"', '')
eval_dir = '//home/developer/Documents/AUC Distracted Driver_split by driver/v2_cam1_cam2_ split_by_driver/Camera 1/Classes_No_TrainVal_Split/c0/Driver 6'


# In[466]:


#load images
eval_imgs = []

for root, dirs, files in os.walk(eval_dir, topdown = True):
    for file in files:
        eval_imgs.append(root + '/' + file)


# In[467]:


#load the model
#Currently loaded: 9312 model
SqueezeNet_Preloaded = tf.keras.models.load_model('//home/developer/Documents/SavedModel')

miss_count = 0
total_count = 0

#predict on a stream of images
for img in eval_imgs:
    eval = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    eval = tf.keras.preprocessing.image.img_to_array(eval)
    eval = np.expand_dims(eval, axis=0)
    
#    with tf.device('/GPU:0'):
    prediction_prob = SqueezeNet_Preloaded.predict(eval)
        
    pred = np.argmax(prediction_prob, axis = 1)[:5] 
    
    total_count += 1
    
    #[0: Distracted Driving, 1: Safe Driving]
    if pred[0] == 0:
        print(pred)
    else:
        print(str(pred) + ' miss')
        miss_count += 1
    break

print(miss_count / total_count * 100)


# In[ ]:


#how many does 9598 miss in validation?
#is the other training method comparable to validation? less or more missed

