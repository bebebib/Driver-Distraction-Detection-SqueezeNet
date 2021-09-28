import tensorflow as tf

#Pass in parameters to create imagedatagenerator objects
def directory_data_gen(rotation, width_shift, height_shift, shear, zoom, fill_mode, preprocess_val):

    #Preprocess training images for better results
    image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator( 
            rotation_range = rotation,  
            width_shift_range = width_shift, 
            height_shift_range = height_shift, 
            shear_range = shear,  
            zoom_range = zoom, 
            fill_mode=fill_mode,
            ) 

    if preprocess_val:
        #No preprocessing on validation images
        image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range = rotation,  
                width_shift_range = width_shift, 
                height_shift_range = height_shift, 
                shear_range = shear,  
                zoom_range = zoom, 
                fill_mode=fill_mode,
                )
    else:
        image_gen_val = tf.keras.preprocessing.image.ImageDataGenerator()

    return image_gen_train, image_gen_val 

#use imagedatagenerator objects and directories to create flow_from_directory objects
def flow_from_directory_gen(image_gen_train, image_gen_val,train_directory, val_directory, batch_size, new_shape, class_mode):
    #Here we load our training set
    flow_from_train = image_gen_train.flow_from_directory(train_directory,
                                                target_size=new_shape,
                                                batch_size=batch_size,
                                                class_mode=class_mode)

    #Here we load our validation set, 
    #which will be used to make sure we are not overfitting the data
    flow_from_val = image_gen_val.flow_from_directory(val_directory,
                                                target_size=new_shape,
                                                batch_size=batch_size,
                                                class_mode=class_mode)

    return flow_from_train, flow_from_val 