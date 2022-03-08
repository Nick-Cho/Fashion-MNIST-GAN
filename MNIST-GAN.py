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
#print("x_train shape: ", x_train.shape)

latent_dim = 100  # latent space dimension

# Generator model


def build_generator(latent_dim):
    i = Input(shape=(latent_dim,))
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.7)(x)
    x = Dense(D, activation='tanh')(x)
    model = Model(i, x)
    return model

# Discriminator model


def build_discriminator(img_size):
    i = Input(shape=(img_size),)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(i, x)
    return model


# building discriminator model
discriminator = build_discriminator(D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy'])

# building generator model
generator = build_generator(latent_dim)

# Creating noise sample and passing into generator
z = Input(shape=(latent_dim,))
img = generator(z)

# Ensure generator is being trained fully before discriminator is trying to decipher image
discriminator.trainable = False

# 1 means picture is fake
fake_pred = discriminator(img)

combined_model = Model(z, fake_pred)

# Compile Combined model to adjust weights
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# training the GAN
