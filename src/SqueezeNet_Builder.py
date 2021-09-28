import tensorflow as tf

#Define a fire_module to add layers to SqueezeNet
def fire_module(x,filter1,filter2,filter3,name):
    
    F_squeeze = tf.keras.layers.Conv2D(filters=filter1, kernel_size=(1,1), kernel_regularizer='l2',padding = 'same', activation='relu', name = 'SqueezeFire' + name)(x)
    F_expand_1x1 = tf.keras.layers.Conv2D(filters=filter2, kernel_size=(1,1), kernel_regularizer='l2', padding = 'same', activation='relu', name = 'Expand1x1Fire' + name)(F_squeeze)
    F_expand_3x3 = tf.keras.layers.Conv2D(filters=filter3, kernel_size=(3,3), kernel_regularizer='l2', padding = 'same', activation='relu', name = 'Expand3x3Fire' + name)(F_squeeze)

    x = tf.keras.layers.Concatenate(axis = -1,name = 'Concatenate' + name)([F_expand_1x1, F_expand_3x3])

    return x

#Build squeezenet and load weights
def build_squeezenet(SqueezeNet, input_layer, weights_fp):
    #Conv2D
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides = (2,2), kernel_regularizer='l2', padding = 'same', activation='relu', name = 'Conv2D_1')(input_layer)
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
    SqueezeNet = tf.keras.Model(input_layer, x, name = 'SqueezeNet')
    
    #Load weights for SqueezeNet trained on ImageNet
    SqueezeNet.load_weights(weights_fp)

    return SqueezeNet

#Modify the last layers of SqueezeNet
def modify_squeezenet(SqueezeNet,input_layer, layer_to_replace, classes):
    hidden = tf.keras.layers.Flatten()(SqueezeNet.layers[layer_to_replace].output)
    hidden = tf.keras.layers.Dense(classes, name = 'DenseFinal', activation = 'softmax')(hidden)

    SqueezeNet_X_Classes = tf.keras.Model(inputs=input_layer, outputs=hidden,name = 'SqueezeNet_' + str(classes) + '_Classes')

    return SqueezeNet_X_Classes