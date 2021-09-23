import tensorflow as tf
import numpy as np
import os, time

#Use GPU
from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#load model and predict
#Currently loaded: Stream Validation 9598
SqueezeNet_Preloaded = tf.keras.models.load_model('tmp/check_point')

val_dir_dist = "Distracted Driver Data_Binary Classification_RandomSelection\\Validation\\Distracted Driving"
val_dir_safe = "Distracted Driver Data_Binary Classification_RandomSelection\\Validation\\Safe Driving"

#load images
eval_imgs_dist = []
eval_imgs_safe = []

for root, dirs, files in os.walk(val_dir_dist, topdown = True):
    for file in files:
        eval_imgs_dist.append(root + '\\' + file)
        
for root, dirs, files in os.walk(val_dir_safe, topdown = True):
    for file in files:
        eval_imgs_safe.append(root + '\\' + file)

time_array_dist = []
time_array_safe = []

#predict on a stream of images
for img in eval_imgs_dist:
    eval = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    eval = tf.keras.preprocessing.image.img_to_array(eval)
    eval = np.expand_dims(eval, axis=0)
    
    with tf.device('/GPU:0'):
        start = time.perf_counter() # more precise
        SqueezeNet_Preloaded.predict(eval)
        end = time.perf_counter #more precise
        time_array_dist.append(end - start)
        
for img in eval_imgs_safe:
    eval = tf.keras.preprocessing.image.load_img(img, target_size=(224, 224))
    eval = tf.keras.preprocessing.image.img_to_array(eval)
    eval = np.expand_dims(eval, axis=0)
    
    with tf.device('/GPU:0'):
        start = time.perf_counter()
        SqueezeNet_Preloaded.predict(eval)
        end = time.perf_counter()
        time_array_safe.append(end - start)


#Calculate final FPS
dist_FPS = sum(time_array_dist)/len(time_array_dist)
safe_FPS = sum(time_array_safe)/len(time_array_safe)
print("Distracted Image FPS: " + dist_FPS)
print("Safe Image FPS: " + safe_FPS)
FPS = (dist_FPS + safe_FPS) / 2
print("Total FPS: " + FPS)




