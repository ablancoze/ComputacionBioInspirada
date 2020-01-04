from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pathlib
import h5py
import cv2
import sys

class dataSet():

    IMG_WIDTH = 3
    IMG_HEIGHT = 3

    def readData(self): 
        data_dir = pathlib.Path('dataset/dataset3/')
        image_count = len(list(data_dir.glob('*.png')))
        print(image_count)
        list_ds = tf.data.Dataset.list_files(str(data_dir/'*'))
        return list_ds

    def __get_label(self,file_path):
        # convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep)
        # The second to last is the class-directory
        return parts[-2] #== self.CLASS_NAMES

    def __decode_img(self,img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    def process_path(self,file_path):
        label = self.__get_label(file_path)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.__decode_img(img)
        return img, label

    def read_data(self, path):
        """
            Read h5 format data file
            Args:
                path: file path of desired file
                data: '.h5' file format that contains  input values
                label: '.h5' file format that contains label values 
        """
        with h5py.File(path, 'r') as hf:
            input_ = np.array(hf.get('input'))
            label_ = np.array(hf.get('label'))
            return input_, label_

    def imread(self, path):
        img = cv2.imread(path)
        return img

    def checkpoint_dir(self, config):
        if config.is_train:
            return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
        else:
            return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

    
    def imsave(self, image, path, config):
        #checkimage(image)
        # Check the check dir, if not, create one
        if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
            os.makedirs(os.path.join(os.getcwd(),config.result_dir))

        # NOTE: because normial, we need mutlify 255 back    
        cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

    def checkimage(self, image):
        cv2.imshow("test",image)
        cv2.waitKey(0)

    def modcrop(self, img, scale =3):
            """
                To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
            """
            # Check the image is grayscale
            if len(img.shape) == 3:
                h, w, _ = img.shape
                h = int(h / scale) * scale
                w = int(w / scale) * scale
                img = img[0:h, 0:w, :]
            else:
                h, w = img.shape
                h = int(h / scale) * scale
                w = int(w / scale) * scale
                img = img[0:h, 0:w]
            return img

    def modcrop_TensorFlow(self, img, scale=3):
        """
            To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
        """
        # Check the image is grayscale
        if len(img.shape) == 3:
            h, w, _ = img.shape
            h = int(h / scale) * scale
            w = int(w / scale) * scale
            img = img[0:h, 0:w, :]
        else:
            h, w = img.shape
            h = int(h / scale) * scale
            w = int(w / scale) * scale
            img = img[0:h, 0:w]
    
        img = tf.image.convert_image_dtype(img,tf.float32)
        return img

    def __prepare_data(self, dataset="Train",Input_img=""):
        """
            Args:
                dataset: choose train dataset or test dataset
                For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
        """
        if dataset == "Train":
            data_dir = os.path.join(os.getcwd(), dataset) # Join the Train dir to current directory
            data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
        else:
            if Input_img !="":
                data = [os.path.join(os.getcwd(),Input_img)]
            else:
                data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Set5")
                data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
        print(data)
        return data


    def load_data(self, is_train, test_img):
        if is_train:
            data = self.__prepare_data(dataset="Train")
        else:
            if test_img != "":
                return self.__prepare_data(dataset="Test",Input_img=test_img)
            data = self.__prepare_data(dataset="Test")
        return data

    def __preprocess(self, path ,scale = 3):
        """
            Args:
                path: the image directory path
                scale: the image need to scale 
        """
        img = self.imread(path)

        label_ = self.modcrop(img, scale)
        
        input_ = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC) # Resize by scaling factor
    
        return input_, label_

    def __preprocessTensorFlow(self, path ,scale = 3):
        """
            Args:
                path: the image directory path
                scale: the image need to scale 
        """
        img = self.imread(path)

        label_ = self.modcrop_TensorFlow(img, scale)

        print("################################################################")
        print("################################################################")
        print(label_)
        print("################################################################")
        print("################################################################")

        input_ = tf.image.resize(label_, [1.0/scale, 1.0/scale], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=False, antialias=False, name=None)
    
        return input_, label_


    def __make_sub_data(self, data, config):
        """
            Make the sub_data set
            Args:
                data : the set of all file path 
                config : the all flags
        """
        sub_input_sequence = []
        sub_label_sequence = []
        for i in range(len(data)):
            input_, label_, = self.__preprocess(data[i], config.scale) # do bicbuic
            if len(input_.shape) == 3: # is color
                h, w, c = input_.shape
            else:
                h, w = input_.shape # is grayscale
            
            if not config.is_train:
                input_ = self.imread(data[i])
                input_ = input_ / 255.0
                sub_input_sequence.append(input_)
                return sub_input_sequence, sub_label_sequence

            # NOTE: make subimage of LR and HR

            # Input 
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):

                    sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 17 * 17


                    # Reshape the subinput and sublabel
                    sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])

                    # Normialize
                    sub_input =  sub_input / 255.0

                    # Add to sequence
                    sub_input_sequence.append(sub_input)

            # Label (the time of scale)
            for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                    sub_label = label_[x: x + config.image_size * config.scale, y: y + config.image_size * config.scale] # 17r * 17r
                    
                    # Reshape the subinput and sublabel
                    sub_label = sub_label.reshape([config.image_size * config.scale , config.image_size * config.scale, config.c_dim])
                    # Normialize
                    sub_label =  sub_label / 255.0
                    # Add to sequence
                    sub_label_sequence.append(sub_label)

        return sub_input_sequence, sub_label_sequence

    def __make_sub_data_TensorFlow(self, data, config):
        """
            Make the sub_data set
            Args:
                data : the set of all file path 
                config : the all flags
        """
        sub_input_sequence = []
        sub_label_sequence = []
        for i in range(len(data)):
            input_, label_, = self.__preprocessTensorFlow(data[i], config.scale) # do bicbuic
            if len(tf.shape(input_)) == 3: # is color
                h, w, c = tf.shape(input_)
            else:
                h, w = tf.shape(input_) # is grayscale
            
            if not config.is_train:
                input_ = self.imread(data[i])
                input_ = input_ / 255.0
                sub_input_sequence.append(input_)
                return sub_input_sequence, sub_label_sequence

            # NOTE: make subimage of LR and HR

            # Input 
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
                    # Reshape the subinput and sublabel
                    sub_input = tf.reshape(input_[x: x + config.image_size, y: y + config.image_size], [config.image_size, config.image_size, config.c_dim])

                    # Normialize
                    sub_input = sub_input / 255.0

                    # Add to sequence
                    sub_input_sequence=tf.concat(sub_input,-1)

            # Label (the time of scale)
            for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                    sub_label = tf.reshape(label_[x: x + config.image_size * config.scale, y: y + config.image_size * config.scale], [config.image_size, config.image_size, config.c_dim])

                    # Normialize
                    sub_label =  sub_label / 255.0
                    # Add to sequence
                    sub_label_sequence=tf.concat(sub_label,-1)

        return sub_input_sequence, sub_label_sequence

    def __make_data_hf(self, input_, label_, config):
        """
            Make input data as h5 file format
            Depending on "is_train" (flag value), savepath would be change.
        """
        # Check the check dir, if not, create one
        if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

        if config.is_train:
            savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
        else:
            savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

        with h5py.File(savepath, 'w') as hf:
            hf.create_dataset('input', data=input_)
            hf.create_dataset('label', data=label_)

    def input_setup(self, config):
        """
            Read image files and make their sub-images and saved them as a h5 file format
        """

        # Load data path, if is_train False, get test data
        data = self.load_data(config.is_train, config.test_img)


        # Make sub_input and sub_label, if is_train false more return nx, ny
        sub_input_sequence, sub_label_sequence = self.__make_sub_data(data, config)


        # Make list to numpy array. With this transform
        arrinput = np.asarray(sub_input_sequence) # [?, 17, 17, 3]
        arrlabel = np.asarray(sub_label_sequence) # [?, 17 * scale , 17 * scale, 3]
        
        print(arrinput.shape)
        self.__make_data_hf(arrinput, arrlabel, config)


    def input_setup_TensorFlow(self, config):
        """
            Read image files and make their sub-images and saved them as a h5 file format
        """

        # Load data path, if is_train False, get test data
        data = self.load_data(config.is_train, config.test_img)


        # Make sub_input and sub_label, if is_train false more return nx, ny
        sub_input_sequence, sub_label_sequence = self.__make_sub_data_TensorFlow(data, config)

        #Falta poder hacer un hd5f a partir de tensores

        # Make list to numpy array. With this transform
        arrinput = np.asarray(sub_input_sequence) # [?, 17, 17, 3]
        arrlabel = np.asarray(sub_label_sequence) # [?, 17 * scale , 17 * scale, 3]
        
        print(arrinput.shape)
        self.__make_data_hf(arrinput, arrlabel, config)