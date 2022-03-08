import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Load in data
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalizing inputs
x_train, x_test = x_train/255.0*2-1, x_test/255.0*2-1
#print("x_train.shape", x_train.shape)

# Flattening Data
N, H, W = x_train.shape  # N-samples, H-height, W-width
D = H*W  # D- number of pixels in the image
x_train, x_test = x_train.reshape(-1, D), x_test.reshape(-1, D)
print("x_train shape: ", x_train.shape)
