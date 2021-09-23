import tensorflow as tf
import numpy as np

#Load model
SqueezeNet_Preloaded = tf.keras.models.load_model('tmp/check_point')

#Apply optimizations and convert
converter = tf.lite.TFLiteConverter.from_keras_model(SqueezeNet_Preloaded)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

#Save model
open("SqueezeNet_Lite_9598.tflite", "wb").write(tflite_model)