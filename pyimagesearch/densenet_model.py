# import the necessary packages
import tensorflow as tf
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class DenseNet():
	def __init__(self, x, params, reuse, is_training):
		self.nb_blocks = 3
		self.params = params
		self.num_filters = 2 * params.growth_rate
		self.reuse = reuse
		self.is_training = is_training
		self.model = Sequential()
		self.inputShape = (self.params.image_size, self.params.image_size, self.params.depth)
		self.chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			self.inputShape = (self.params.depth, self.params.image_size, self.params.image_size)
			self.chanDim = 1

		self.Dense_net(x, self.inputShape)

	def transition_layer(self, x, scope):
		with tf.name_scope(scope):
			x = tf.layers.batch_normalization(x, momentum=self.params.bn_momentum, training=self.is_training)
			x = tf.nn.relu(x)
			num_output_channels = int(self.num_filters * self.params.compression_rate)
			x = conv_layer(x, num_output_channels, kernel=[1, 1], layer_name=scope + '_conv1')
			if self.params.dropout_rate > 0:
				x = tf.layers.dropout(x, rate=self.params.dropout_rate, training=self.is_training)
			x = Average_pooling(x, pool_size=[2, 2], stride=2)
			return x

	def bottleneck_layer(self, no_filters, scope):
		with tf.name_scope(scope):
			self.model.add(BatchNormalization(axis=self.chanDim))
			self.model.add(Activation("relu"))
			num_channels = no_filters * 4
			self.model.add(Conv2D(num_channels, (1, 1), padding="same"))
			if self.params.dropout_rate > 0:
				self.model.add(Dropout(self.params.dropout_rate))
			self.model.add(MaxPooling2D(pool_size=(3, 3), stride=2))
			self.model.add(BatchNormalization(axis=self.chanDim))
			self.model.add(Activation("relu"))
			self.model.add(Conv2D(num_channels, (3, 3), padding="same"))
			if self.params.dropout_rate > 0:
				self.model.add(Dropout(self.params.dropout_rate))

	def dense_block(self, input_x, nb_layers, layer_name):
		with tf.name_scope(layer_name):
			concat_feat = input_x
			for i in range(nb_layers):
				x = self.bottleneck_layer(concat_feat, no_filters=self.params.growth_rate,
										  scope=layer_name + '_bottleN_' + str(i + 1))
				concat_feat = tf.concat([concat_feat, x], axis=3)
				self.num_filters += self.params.growth_rate
			return concat_feat

	def Dense_net(self, input_x):


		with tf.variable_scope('DenseNet-v', reuse=self.reuse):
			self.model.add(Conv2D(self.num_filters, (7, 7), stride=2, padding="same",
							 input_shape=self.inputShape))
			self.model.add(BatchNormalization(axis=self.chanDim))
			self.model.add(Activation("relu"))
			self.model.add(MaxPooling2D(pool_size=(3, 3), stride=2))
			self.model.add(Dropout(0.25))

			# define list contain the number layers in blocks the length of list based on the number blocks in the model

			out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[0], layer_name='dense_1')
			out = self.transition_layer(out, scope='trans_1')
			self.num_filters = int(self.num_filters * self.params.compression_rate)

			out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[1], layer_name='dense_2')
			out = self.transition_layer(out, scope='trans_2')
			self.num_filters = int(self.num_filters * self.params.compression_rate)

			out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[2], layer_name='dense_3')
			out = self.transition_layer(out, scope='trans_3')
			self.num_filters = int(self.num_filters * self.params.compression_rate)

			out = self.dense_block(input_x=out, nb_layers=self.params.num_layers_per_block[3], layer_name='dense_4')
			out = tf.layers.batch_normalization(out, momentum=self.params.bn_momentum, training=self.is_training)
			out = tf.nn.relu(out)
			out = Global_Average_Pooling(out)

			# num_layers_in_block = [1, 1, 1]
			# for i in range(len(num_layers_in_block)):
			#     # 6 -> 12 -> 48
			#     out = self.dense_block(input_x=out, nb_layers=int(num_layers_in_block[i]), layer_name='dense_'+str(i))
			#     out = self.transition_layer(out, scope='trans_'+str(i))
			#     self.num_filters = int(self.num_filters * self.params.compression_rate)
			# x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

			# 100 Layer
			'''
            # out = tf.reshape(out, [-1, 1 * 1 * self.num_filters])
            with tf.variable_scope('fc_1'):
                fc1 = tf.layers.dense(out, self.num_filters)
                # Apply Dropout (if is_training is False, dropout is not applied)
                if self.params.dropout_rate >0:
                    fc1 = tf.layers.dropout(fc1, rate=self.params.dropout_rate, training=self.is_training)
                # if self.params.use_batch_norm:
                #     out = tf.layers.batch_normalization(out, momentum=self.params.bn_momentum, training=self.is_training)
                fc1 = tf.nn.relu(fc1)
            '''
			with tf.variable_scope('fc_1'):
				fc1 = tf.layers.flatten(out)
			with tf.variable_scope('fc_2'):
				logits = tf.layers.dense(fc1, self.params.num_labels)



	# Because 'softmax_cross_entropy_with_logits' already apply softmax,
	# we only apply softmax to testing network
	# x = tf.nn.softmax(x) if not self.is_training else x


