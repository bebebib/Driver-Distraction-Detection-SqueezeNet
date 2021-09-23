
import tensorflow as tf
import numpy as np
import os
from shutil import rmtree
from tensorflow.python.client import device_lib

#Use GPU for deep learning
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#Load directories
train_directory = input("Where is the training data: ")
train_directory = train_directory.replace('"', '')

val_directory = input("Where is the validation data: ")
val_directory = val_directory.replace('"', '')

#Preprocess training images for better results
image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator( 
        rotation_range=30, # rotate the image 30 degrees   
        width_shift_range=0.1, # Shift the pic width by a max of 10%
        height_shift_range=0.1, # Shift the pic height by a max of 10%
        shear_range=0.2, # Shear means cutting away part of the image (max 20%)
        zoom_range=0.2, # Zoom in by 20% max
        fill_mode='nearest',#Fill in missing pixels with the nearest filled value
        ) 

#No preprocessing on validation images
image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator() 

#Here we define batch size, as well as the reshaping of our image
batch_size_train = 64
batch_size_val = batch_size_train
new_shape = [224, 224] #Input image size from SqueezeNet paper

#Here we load our training set
train_image_gen = image_gen_train.flow_from_directory(train_directory,
                                               target_size=new_shape,
                                               batch_size=batch_size_train,
                                               class_mode='categorical')

#Here we load our validation set, 
#which will be used to make sure we are not overfitting the data
val_image_gen = image_gen_val.flow_from_directory(val_directory,
                                               target_size=new_shape,
                                               batch_size=batch_size_val,
                                               class_mode='categorical')

#Define a fire_module to add layers to SqueezeNet
def fire_module(x,filter1,filter2,filter3,name):
    
    F_squeeze = tf.keras.layers.Conv2D(filters=filter1, kernel_size=(1,1), kernel_regularizer='l2',padding = 'same', activation='relu', name = 'SqueezeFire' + name)(x)
    F_expand_1x1 = tf.keras.layers.Conv2D(filters=filter2, kernel_size=(1,1), kernel_regularizer='l2', padding = 'same', activation='relu', name = 'Expand1x1Fire' + name)(F_squeeze)
    F_expand_3x3 = tf.keras.layers.Conv2D(filters=filter3, kernel_size=(3,3), kernel_regularizer='l2', padding = 'same', activation='relu', name = 'Expand3x3Fire' + name)(F_squeeze)

    x = tf.keras.layers.Concatenate(axis = -1,name = 'Concatenate' + name)([F_expand_1x1, F_expand_3x3])

    return x



#Here we build Squeezenet v1.1

#Re-intialize
SqueezeNet = 0
x = 0
img_input = tf.keras.Input(shape=(224,224,3), name = 'Input')
#Conv2D
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides = (2,2), kernel_regularizer='l2', padding = 'same', activation='relu', name = 'Conv2D_1')(img_input)
#Max Pool
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides = (2,2), padding = 'valid', name = 'MaxPool1')(x)
#Fire 2
x = fire_module(x,16,64,64,'2')
#Fire 3
x = fire_module(x,16,64,64,'3')
#Fire 4
x = fire_module(x,32,128,128,'4')
#Max Pool
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides = (2,2), name = 'MaxPool4')(x)
#Fire 5
x = fire_module(x,32,128,128,'5')
#Fire 6
x = fire_module(x,48,192,192,'6')
#Fire 7
x = fire_module(x,48,192,192,'7')
#Fire 8
x = fire_module(x,64,256,256,'8')
#Max Pool
x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides = (2,2), name = 'MaxPool8')(x)
#Fire 9
x = fire_module(x,64,256,256,'9')
#Dropout 
x = tf.keras.layers.Dropout(0.5, name = 'Dropout9')(x)
#Conv2D
x = tf.keras.layers.Conv2D(filters=1000, kernel_size=(1,1), strides = (1,1), padding = 'same', activation='relu', name = 'Conv2D_10')(x)
#Max Pool
x = tf.keras.layers.AveragePooling2D(pool_size=(13, 13), strides = (1,1), name = 'MaxPool10')(x)
SqueezeNet = tf.keras.Model(img_input, x, name = 'SqueezeNet')
SqueezeNet.summary()

#Load weights for SqueezeNet trained on ImageNet
SqueezeNet.load_weights("squeezenet_v1.1_weights.h5")

#Here the last layer of SqueezeNet is replaced with a Dense Layer of two neurons, since this is a binary classification
SqueezeNet_Preloaded = 0
hidden = 0

hidden = tf.keras.layers.Flatten()(SqueezeNet.layers[-3].output)
hidden = tf.keras.layers.Dense(2, name = 'DenseFinal', activation = 'softmax')(hidden)

SqueezeNet_Preloaded = tf.keras.Model(inputs=img_input, outputs=hidden,name = 'SqueezeNet_Preloaded')

SqueezeNet_Preloaded.summary()

#Optimizers and model compilation
opt = tf.keras.optimizers.SGD(lr=0.01, decay=0.0001, clipnorm=1)

SqueezeNet_Preloaded.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


#As we train the model, we save the model with the highest validation accuracy
checkpoint_filepath = 'tmp/check_point'

restart_model_save = input("Would you like to train a new model? (y/n): ")

if restart_model_save == 'n':
    print("Picking up where left off")
    SqueezeNet_Preloaded = tf.keras.models.load_model('tmp/check_point')
elif restart_model_save == 'y':
    print("Previous model deleted")
    rmtree(checkpoint_filepath)
    os.makedirs(checkpoint_filepath)

#Checkpoing call back definition
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

#Train SqueezeNet using desktop GPU
with tf.device('/GPU:0'):
    results = SqueezeNet_Preloaded.fit_generator(train_image_gen,
                                   epochs=50,
                              steps_per_epoch=len(train_image_gen.filenames) / batch_size_train,
                             validation_data=val_image_gen,
                             validation_steps=len(val_image_gen.filenames) / batch_size_val,
                             callbacks=[model_checkpoint_callback])





