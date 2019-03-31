# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

from feed_data_seq import Data_Generator


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

def get_labels(imagePaths):
	labels = []
	for imagePath in imagePaths:
		# extract the class label from the image path and update the
		# labels list
		label = imagePath.split(os.path.sep)[-2]
		labels.append(label)
	return labels
# /home/ai309/metwalli/project-test-1/dense_food/experiments/vireo10_aug4
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 10
INIT_LR = 1e-3
BATCH_SIZE = 16
IMAGE_DIMS = (64, 64, 3)
N_CLASSES = 10
# initialize the data and labels
data = []
train_labels = []
eval_labels = []
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
train_filenames = sorted(list(paths.list_images(os.path.join(args["dataset"], "train"))))
random.seed(42)
random.shuffle(train_filenames)
train_labels = get_labels(train_filenames)

eval_filenames = sorted(list(paths.list_images(os.path.join(args["dataset"], "test"))))
eval_labels = get_labels(eval_filenames)

my_training_batch_generator = Data_Generator(image_filenames=train_filenames, labels=train_labels, batch_size=BATCH_SIZE)
my_validation_batch_generator = Data_Generator(image_filenames=eval_filenames, labels=eval_labels, batch_size=BATCH_SIZE)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build2(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=N_CLASSES)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch=(len(train_filenames) // BATCH_SIZE),
                                          epochs=EPOCHS,
                                          verbose=1,
                                          validation_data=my_validation_batch_generator,
                                          validation_steps=(len(eval_filenames) // BATCH_SIZE))

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])