import pathlib
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import tensorflow as tf
from base64 import b64decode
from inputConstants import *

class CNNModel():
	def __init__(self):
		vggNet = tf.keras.applications.VGG16(
			input_shape=(IMG_WIDTH,IMG_HEIGHT,CHANNELS),
			include_top = False,
			weights = 'imagenet')
		vggNet.trainable = False
		self.myModel = tf.keras.Sequential([
			vggNet,
			tf.keras.layers.GlobalAveragePooling2D(),
			tf.keras.layers.Dropout(0.5),
			tf.keras.layers.Dense(len(LABELS),activation=tf.nn.softmax)
		])
		checkpoint_path = "./checkpoint_trainingVGG/cp-{epoch:04d}.ckpt"
		checkpoint_dir = os.path.dirname(checkpoint_path)
		self.myModel.compile(optimizer=tf.train.AdamOptimizer(learning_rate=1e-4),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=["accuracy"])
		latest = tf.train.latest_checkpoint(checkpoint_dir)
		self.myModel.load_weights(latest)
    
	def predict(self,image):
		pred = self.myModel.predict_classes(self._convertPngImage(image))
		return imgClassDict['c'+ str(pred[0])]
	def _preprocessImage(self,image):
		#image = tf.image.decode_png(image, channels=CHANNELS)
		image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
		image /= 255.0  # normalize to [0,1] range
		return tf.expand_dims(image,0)
    
	def _loadImage(self,image):
		return self._preprocessImage(image)

	def _convertPngImage(self,image):
		header, encoded = image.split(",", 1)
		data = b64decode(encoded)
		return self._loadImage(tf.io.decode_image(data,channels=3))



