#-------------------------------#

from tensorflow.keras.layers import Input, ConvLSTM2D, Concatenate, Dropout,\
    TimeDistributed, SeparableConv2D, GlobalAveragePooling2D, Dense, GlobalAveragePooling3D, MaxPooling2D
from sklearn.metrics import classification_report
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
from itertools import chain
from random import shuffle
import numpy as np
import glob
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm

# This is where TrainWeather.py is located
working_dir = "/homes/enueve/WeatherNet/UploadWeatherNet"
# This is where the folder data_npy is located
data_dir = '/homes/enueve/WeatherNet/UploadWeatherNet/data_npy'

#-------------------------------#

train_path = data_dir + "/train"
val_path = data_dir + "/val"
test_path = data_dir + "/test"

train_flir = train_path + "/flir"
train_top = train_path + "/top"
train_bottom = train_path + "/bottom"

val_flir = val_path + "/flir"
val_top = val_path + "/top"
val_bottom = val_path + "/bottom"

test_flir = test_path + "/flir"
test_top = test_path + "/top"
test_bottom = test_path + "/bottom"

#-------------------------------#


def get_dataset(flir_path, bottom_path, top_path):
    files = [os.path.basename(file_name) for i, file_name in enumerate(
        glob.glob(flir_path+"/*.npy"))]

    X_flir = np.empty((len(files), *(6, 3, 480, 640)))
    X_bottom = np.empty((len(files), *(6, 3, 480, 640)))
    X_top = np.empty((len(files), *(6, 3, 480, 640)))
    y = np.empty(len(files), dtype=int)
    # Generate data
    for i, ID in tqdm(enumerate(files)):
        # Store sample
        X_flir[i, ] = np.load(flir_path + '/' + ID)
        X_bottom[i, ] = np.load(bottom_path + '/' + ID)
        X_top[i, ] = np.load(top_path + '/' + ID)

    labels = [file.split("_")[-1] for i, file in enumerate(files)]
    labels = [file.split(".")[0] for i, file in enumerate(labels)]
    for i, index in enumerate(labels):
        # Store class
        if index == "low":
            y[i] = 0
        elif index == "mid":
            y[i] = 1
        else:
            y[i] = 2

    return [X_flir/255.0, X_bottom/255.0, X_top/255.0], to_categorical(y, num_classes=3)


X_train, y_train = get_dataset(train_flir, train_bottom, train_top)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
input("press enter")
