from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from IPython import display

class Subpixel(object):
    def __init__(self,sess,image_size,is_train,scale,batch_size,c_dim,test_img,):
        self.sess = sess
        self.image_size = image_size
        self.is_train = is_train
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.test_img = test_img
        self.build_model()

    def build_model(self):

        if self.is_train:
            self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.image_size * self.scale , self.image_size * self.scale, self.c_dim], name='labels')
        else:
            '''
                Because the test need to put image to model,
                so here we don't need do preprocess, so we set input as the same with preprocess output
            '''
            data = load_data(self.is_train, self.test_img)
            input_ = imread(data[0])       
            self.h, self.w, c = input_.shape
            self.images = tf.placeholder(tf.float32, [None, self.h, self.w, self.c_dim], name='images')
            self.labels = tf.placeholder(tf.float32, [None, self.h * self.scale, self.w * self.scale, self.c_dim], name='labels')
        
        self.weights = {
            'w1': tf.Variable(tf.random_normal([5, 5, self.c_dim, 64], stddev=np.sqrt(2.0/25/3)), name='w1'),
            'w2': tf.Variable(tf.random_normal([3, 3, 64, 32], stddev=np.sqrt(2.0/9/64)), name='w2'),
            'w3': tf.Variable(tf.random_normal([3, 3, 32, self.c_dim * self.scale * self.scale ], stddev=np.sqrt(2.0/9/32)), name='w3')
        }

        self.biases = {
            'b1': tf.Variable(tf.zeros([64], name='b1')),
            'b2': tf.Variable(tf.zeros([32], name='b2')),
            'b3': tf.Variable(tf.zeros([self.c_dim * self.scale * self.scale ], name='b3'))
        }
        
        self.pred = self.model()
        
        self.loss = tf.reduce_mean(tf.square(self.labels - self.pred))

        self.saver = tf.train.Saver() # To save checkpoint

    def _phase_shift(self, I, r):
        # Helper function with main phase shift operation
        bsize, a, b, c = I.get_shape().as_list()
        X = tf.reshape(I, (self.batch_size, a, b, r, r))
        X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
        X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
        X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
        return tf.reshape(X, (self.batch_size, a*r, b*r, 1))

    def periodicShuffling (self, X, r):
        # Main OP that you can arbitrarily use in you tensorflow code
        Xc = tf.split(X, 3, 3)
        if self.is_train:
            X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
        return X
    
    def model(self):
        conv1 = tf.nn.relu(tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='SAME') + self.biases['b1'])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='SAME') + self.biases['b2'])
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='SAME') + self.biases['b3'] # This layer don't need ReLU

        ps = self.PS(conv3, self.scale)
        return tf.nn.tanh(ps)