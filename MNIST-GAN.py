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
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
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

gan = Model(z, fake_pred)

# Compile Combined model to adjust weights
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Training the GAN

batch_size = 32
epochs = 30000
sample_period = 200  # data is saved every time the sample period passes

# Creating Batch Labels
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

d_losses = []  # discriminator loss
g_losses = []  # generator loss

# Creating folder to store images
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')


def sample_images(epoch):
    rows, cols = 5, 5
    # creating noise to input into the generator
    noise = np.random.randn(rows*cols, latent_dim)
    imgs = generator.predict(noise)

    # Rescale images to 0-1
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[idx].reshape(H, W), cmap='gray')
            axs[i, j].axis('off')
            idx += 1
    fig.savefig("gan_images/%d.png" % epoch)
    plt.close()


for epoch in range(epochs):
    # Training discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    noise = np.random.randn(batch_size, latent_dim)
    fake_imgs = generator.predict(noise)

    d_loss_real, d_acc_real = discriminator.train_on_batch(real_imgs, ones)
    d_loss_fake, d_acc_fake = discriminator.train_on_batch(fake_imgs, zeros)
    d_loss = 0.5 * (d_loss_real + d_loss_fake)
    d_acc = 0.5 * (d_acc_real + d_acc_fake)

    # Training generator
    noise = np.random.randn(batch_size, latent_dim)
    g_loss = gan.train_on_batch(noise, ones)

    noise = np.random.randn(batch_size, latent_dim)
    g_loss = gan.train_on_batch(noise, ones)

    # Save the losses
    d_losses.append(d_loss)
    g_losses.append(g_loss)

    if epoch % 100 == 0:
        print(f"epoch: {epoch+1}/{epochs}, d_loss: {d_loss:.2f}, \
      d_acc: {d_acc:.2f}, g_loss: {g_loss:.2f}")

    if epoch % sample_period == 0:
        sample_images(epoch)
