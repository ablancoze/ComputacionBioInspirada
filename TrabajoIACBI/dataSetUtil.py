from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib


class dataSet():

  IMG_WIDTH = 3
  IMG_HEIGHT = 3
  CLASS_NAMES

  def readData(self): 
    data_dir = pathlib.Path('dataset/dataset3/')
    image_count = len(list(data_dir.glob('*.png')))
    print(image_count)
    list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))


  def get_label(self,file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    return parts[-2] #== self.CLASS_NAMES

  def decode_img(self,img):
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

  def process_path(self,file_path):
    label = self.get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = self.decode_img(img)
    return img, label

data_dir = pathlib.Path('dataset/dataset3/')
image_count = len(list(data_dir.glob('*.png')))

print(image_count)

list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))

for f in list_ds:
  img=f.numpy()
  print(type(img))

print(type(list_ds))