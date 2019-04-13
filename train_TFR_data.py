
# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background

import tensorflow as tf
import matplotlib
import time
matplotlib.use("Agg")
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.objectives import categorical_crossentropy
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

from tensorflow.python.keras.callbacks import TensorBoard
from input_fn import input_fn
from pyimagesearch.densenet_model import densenet121_model
from pyimagesearch.smallervggnet import SmallerVGGNet
from pyimagesearch.vgg16 import VGGNet16

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train_tf", required=True,
                help="path to train TFRecord file")
ap.add_argument("-e", "--eval_tf", required=True,
				help="Evalu TFRecord File")
ap.add_argument("-c", "--ckpt_dir", required=True,
                help="path of check points (i.e., directory of check points)")
ap.add_argument("-r", "--restore_from", required=False,
                help="path of saved checkpoints (i.e., directory of check points)")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
                help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
# /home/ai309/metwalli/project-test-1/dense_food/experiments/vireo10_aug4
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 10
INIT_LR = 1e-4
BS = 1
CLASSES = 5
BATCH_SHAPE = [BS, 224, 224, 3]
PARALLELISM = 4

use_pretrained = True


# grab the train image paths and randomly shuffle them
print("[INFO] loading images...")
train_tf = args["train_tf"]
eval_tf = args["eval_tf"]

train_size = len([x for x in tf.python_io.tf_record_iterator(train_tf)])
eval_size = len([x for x in tf.python_io.tf_record_iterator(eval_tf)])
print(train_size)
x_train_batch, y_train_batch = input_fn(
    train_tf,
    one_hot=True,
    classes=CLASSES,
    is_training=True,
    batch_shape=BATCH_SHAPE,
    parallelism=PARALLELISM)
x_test_batch, y_test_batch = input_fn(
    eval_tf,
    one_hot=True,
    classes=CLASSES,
    is_training=True,
    batch_shape=BATCH_SHAPE,
    parallelism=PARALLELISM)

x_batch_shape = x_train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()

x_train_input = Input(tensor=x_train_batch, batch_shape=x_batch_shape)
y_train_in_out = Input(tensor=y_train_batch, batch_shape=y_batch_shape, name='y_labels')


x_train_out = densenet121_model(img_input=x_train_input, use_pretrained=use_pretrained, num_classes=CLASSES)
cce = categorical_crossentropy(y_train_batch, x_train_out)
model = Model(inputs=[x_train_input], outputs=[x_train_out])
model.add_loss(cce)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
# model = VGGNet16.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], depth=IMAGE_DIMS[2], classes=5)
# Load our model
# model = densenet_model(img_rows=IMAGE_DIMS[0], img_cols=IMAGE_DIMS[1], color_type=IMAGE_DIMS[2], num_classes=len(lb.classes_))
# model = densenet121_model(img_rows=IMAGE_DIMS[0], img_cols=IMAGE_DIMS[1], color_type=IMAGE_DIMS[2], num_classes=len(lb.classes_))

# tensorBoard = TensorBoard(log_dir='logs/{}'.format(time.time()))
# # checkpoint
filepath= os.path.join(args["ckpt_dir"], "weights.best.hdf5")
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = checkpoint
# if args["restore_from"] is not None:
#     if os.path.isdir(args["restore_from"]):
#         model.load_weights(filepath)


# train the network
print("[INFO] training network...")
optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
# optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=optimizer, metrics=["accuracy"])

model.summary()

tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))
H = model.fit(epochs=EPOCHS,
          steps_per_epoch=train_size//BS,
          callbacks=[checkpoint, tensorboard])

model.save_weights('saved_wt.h5')

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.subplot(1)
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.subplot(2)
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])