import cv2
from scipy.ndimage import zoom
import os
import pathlib
import time
import datetime
import sys
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def convertIntBin(array):
    (array > 0.5).astype(np.bool_)

def reshape(img):
    img = np.expand_dims(img, axis=0) 
    img = np.expand_dims(img, axis=-1)
    img = np.reshape(img, (1, img.shape[1], img.shape[2],img.shape[3]))
    return img

def reshape3D(array):
    return zoom(array, (0.25,0.25,0.25))


#as far as I did understand he's minipulating the training data for better trainning and to overcome the overfitting problem
def load_image_train(image_file):
  input_image = load(image_file)
  input_image = random_jitter(input_image)

  return input_image
# Define a function to load an image and binvox file
def load_data(image_path, binvox_path):
    # Load the image file
    image = tf.io.decode_png(tf.io.read_file(image_path), channels=1)
    
    image = resize(image, 128,128)
    
    # Load the binvox file
    #I need to get the name of the binvox file corresponds to the img file and return it

    vox = load3DModel(binvox_path)

    # Convert the voxels to a binary array
    voxels = (vox.data > 0).astype(np.int32)
    voxels = reshape3D(voxels)

    return image, voxels


def load_image_test(image_file):
  input_image = load(image_file)
  input_image = resize(input_image,IMG_HEIGHT, IMG_WIDTH)

  return input_image


def random_jitter(input_image):
  # Resizing to 286x286
  input_image = resize(input_image,IMG_WIDTH ,IMG_HEIGHT )

  return input_image


# Normalizing the images to [-1, 1]
def normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image


def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return input_image
    
def imgToGrey(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

#he's going to create a fn to load the images and splite them into 2 images 
def load(image_file):
    image = tf.io.decode_png(tf.io.read_file(image_file), channels=1)
  # Convert both images to float32 tensors
    return image