#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import os
import tensorflow as tf
import time


# In[2]:


from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# In[3]:


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="//home/developer/Documents/tfliteModel/SqueezeNet_Lite_9598.tflite",num_threads=4)
interpreter.allocate_tensors()


# In[4]:


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[5]:


# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()


# In[6]:


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


# In[14]:


val_dir_dist = "//home/developer/Documents/Validation/Distracted Driving"
val_dir_safe = "//home/developer/Documents/Validation/Safe Driving"


# In[15]:


#load images
eval_imgs_dist = []
eval_imgs_safe = []

for root, dirs, files in os.walk(val_dir_dist, topdown = True):
    for file in files:
        eval_imgs_dist.append(root + '/' + file)
        
for root, dirs, files in os.walk(val_dir_safe, topdown = True):
    for file in files:
        eval_imgs_safe.append(root + '/' + file)


# In[34]:


time_array_dist = []
time_array_safe = []

#predict on a stream of images
for img in eval_imgs_dist:
    eval = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    eval = tf.keras.preprocessing.image.img_to_array(eval)
    eval = np.expand_dims(eval, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], eval)
    
    with tf.device('/GPU:0'):
        start = time.time()
        interpreter.invoke()
        end = time.time()
        time_array_dist.append(end - start)
    
    print(end - start)

for img in eval_imgs_safe:
    eval = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    eval = tf.keras.preprocessing.image.img_to_array(eval)
    eval = np.expand_dims(eval, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], eval)
    
    with tf.device('/GPU:0'):
        start = time.time()
        interpreter.invoke()
        end = time.time()
        time_array_safe.append(end - start)
   
    print(end - start)

# In[21]:


dist_FPS = sum(time_array_dist)/len(time_array_dist)
safe_FPS = sum(time_array_safe)/len(time_array_safe)
print("Distracted FPS: "+ str(dist_FPS))
print("Safe FPS: " + str(safe_FPS))
FPS = (dist_FPS + safe_FPS) / 2
print("The final FPS: " + str(FPS))

