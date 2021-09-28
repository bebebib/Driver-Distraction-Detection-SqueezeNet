import os
import tensorflow as tf
from shutil import rmtree
from tensorflow.python.client import device_lib
from Dataset_Generator import directory_data_gen, flow_from_directory_gen
from SqueezeNet_Builder import build_squeezenet, modify_squeezenet

#Use GPU for deep learning
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

if __name__ == '__main__':
    #Local variables
    SqueezeNet = 0
    batch_size = 64
    epochs = 50
    img_size = [224, 224]

    #Load available GPU
    print(get_available_devices()) 
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    #Load directories
    train_directory = input("Where is the training data: ")
    train_directory = train_directory.replace('"', '')
    val_directory = input("Where is the validation data: ")
    val_directory = val_directory.replace('"', '')

    #Create Image data generators, no preprocessing on validation data
    image_gen_train, image_gen_val = directory_data_gen(30, 0.1, 0.1, 0.2, 0.2, "nearest", False)

    #Flow from directory for each image data generator
    flow_from_train, flow_from_val = flow_from_directory_gen(image_gen_train,image_gen_val,train_directory, val_directory, batch_size, img_size, "categorical")

    weights_fp = input("What is the path of the SqueezeNet weights: ")
    weights_fp = weights_fp.replace('"', '')

    #Input layer
    img_input = tf.keras.Input(shape=(224,224,3), name = 'Input')    
    
    #Build SqueezeNet
    SqueezeNet = build_squeezenet(SqueezeNet, img_input, weights_fp)

    #Modify SqueezeNet for a binary classification
    SqueezeNet_2_Classes = modify_squeezenet(SqueezeNet,img_input, -3, 2)

    #Optimizers and model compilation
    opt = tf.keras.optimizers.SGD(lr=0.01, decay=0.0001, clipnorm=1)
    SqueezeNet_2_Classes.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

    #As we train the model, we save the model with the highest validation accuracy
    checkpoint_filepath = 'tmp/check_point'

    restart_model_save = input("Would you like to train a new model? (y/n): ")

    if restart_model_save == 'n':
        print("Picking up where left off")
        SqueezeNet_2_Classes = tf.keras.models.load_model('tmp/check_point')
    else:
        print("Previous model deleted")
        try: 
            rmtree(checkpoint_filepath)
        except:
            print("Nothing to delete, making files")
        os.makedirs(checkpoint_filepath)
    
    #Checkpoint call back definition
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    #Train SqueezeNet using desktop GPU
    with tf.device('/GPU:0'):
        results = SqueezeNet_2_Classes.fit_generator(flow_from_train,
                                    epochs=epochs,
                                    steps_per_epoch=len(flow_from_train.filenames) / batch_size,
                                    validation_data=flow_from_val,
                                    validation_steps=len(flow_from_val.filenames) / batch_size,
                                    callbacks=[model_checkpoint_callback])
