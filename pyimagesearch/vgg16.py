# import the necessary packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class VGGNet16:
	@staticmethod
	def build(width, height, depth, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		model_input = keras.layers.Input(inputShape)
		# CONV => RELU => POOL
		out = keras.layers.Conv2D(32,(3, 3), padding="same", activation="relu")(model_input)
		out = keras.layers.BatchNormalization(axis=chanDim)(out)
		out = keras.layers.MaxPool2D(pool_size=(3, 3))(out)
		out = keras.layers.Dropout(0.25)(out)

		# (CONV => RELU) * 2 => POOL
		out = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(out)
		out = keras.layers.BatchNormalization(axis=chanDim)(out)
		out = keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(out)
		out = keras.layers.BatchNormalization(axis=chanDim)(out)
		out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)
		out = keras.layers.Dropout(0.25)(out)

		# (CONV => RELU) * 2 => POOL
		out = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(out)
		out = keras.layers.BatchNormalization(axis=chanDim)(out)
		out = keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu")(out)
		out = keras.layers.BatchNormalization(axis=chanDim)(out)
		out = keras.layers.MaxPool2D(pool_size=(2, 2))(out)
		out = keras.layers.Dropout(0.25)(out)

		# first (and only) set of FC => RELU layers
		# out = tf.reshape(out, 5 * 5 * 128)
		# out = keras.layers.Flatten()(out)
		out = keras.layers.Dense(1024, activation="relu")(out)
		out = keras.layers.BatchNormalization()(out)
		out = keras.layers.Dropout(0.5)(out)
		model_output = keras.layers.Dense(classes, activation="softmax")(out)
		model = keras.models.Model(model_input, model_output)

		return model


		# # CONV => RELU => POOL
		# model.add(Conv2D(32, (3, 3), padding="same",
		# 	input_shape=inputShape))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(3, 3)))
		# model.add(Dropout(0.25))
		#
		# # (CONV => RELU) * 2 => POOL
		# model.add(Conv2D(64, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(Conv2D(64, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))
		#
		# # (CONV => RELU) * 2 => POOL
		# model.add(Conv2D(128, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(Conv2D(128, (3, 3), padding="same"))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization(axis=chanDim))
		# model.add(MaxPooling2D(pool_size=(2, 2)))
		# model.add(Dropout(0.25))
		#
		# # first (and only) set of FC => RELU layers
		# model.add(Flatten())
		# model.add(Dense(1024))
		# model.add(Activation("relu"))
		# model.add(BatchNormalization())
		# model.add(Dropout(0.5))
		#
		# # softmax classifier
		# model.add(Dense(classes))
		# model.add(Activation("softmax"))
		#
		# # return the constructed network architecture
		# return model

	def build2(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		num_channels = 64
		bn_momentum = 0.9
		model.add(Conv2D(num_channels, (3, 3), padding="same", input_shape=inputShape))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		channels = [num_channels * 2, num_channels * 4, num_channels * 8]
		for i, c in enumerate(channels):
			model.add(Conv2D(c, (3, 3), padding="same"))
			model.add(BatchNormalization(axis=chanDim))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dense(8192))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Activation("relu"))
		model.add(Dense(classes))

		return model