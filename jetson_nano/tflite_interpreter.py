#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import os
import tensorflow as tf


# In[21]:


from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# In[26]:


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="//home/developer/Documents/tfliteModel/SqueezeNet_Lite_9598.tflite")
interpreter.allocate_tensors()


# In[27]:


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# In[28]:


# Test the model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()


# In[29]:


# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


# In[30]:


# Training - All classes
# [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 28, 29]
# Validation - All classes
# [2, 12, 30, 21, 14, 27]

eval_dir = "//home/developer/Documents/AUC Distracted Driver_split by driver/v2_cam1_cam2_ split_by_driver/Camera 1/Classes_No_TrainVal_Split/c0/Driver 27"


# In[31]:


#load images
eval_imgs = []

for root, dirs, files in os.walk(eval_dir, topdown = True):
    for file in files:
        eval_imgs.append(root + '/' + file)


# In[32]:


miss_count = 0
total_count = 0

#predict on a stream of images
for img in eval_imgs:
    eval = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    eval = tf.keras.preprocessing.image.img_to_array(eval)
    eval = np.expand_dims(eval, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], eval)
    
    with tf.device('/GPU:0'):
        interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    pred = np.argmax(output_data, axis = 1)[:5] 
    
    total_count += 1
    
    #[0: Distracted Driving, 1: Safe Driving]
    if pred[0] == 1:
        print(pred)
    else:
        print(str(pred) + ' miss')
        miss_count += 1

print(miss_count / total_count * 100)


# In[ ]:




