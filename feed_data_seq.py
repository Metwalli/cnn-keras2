from skimage.io import imread
from skimage.transform import resize
import keras.utils as utils
import tensorflow as tf
# from tensorflow.keras.utils import Sequence
import numpy as np

class Data_Generator(utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, dim=(32,32,32), n_channels=1,
             n_classes=10, shuffle=True):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID)

            # Store class
            y[i] = self.labels[ID]

        return X, utils.to_categorical(y, num_classes=self.n_classes)

    def __len__(self):
        return np.ceil(len(self.image_filenames) / float(self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x], dtype="float")/255.0, np.array(batch_y, dtype=int)
