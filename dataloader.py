import os
from os.path import isdir, exists, abspath, join
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

class DataLoader():
    def __init__(self, root_dir='data', batch_size=2, test_percent=.1):
        self.batch_size = batch_size
        self.test_percent = test_percent

        self.root_dir = abspath(root_dir)
        self.data_dir = join(self.root_dir, 'scans')
        self.labels_dir = join(self.root_dir, 'labels')

        self.files = os.listdir(self.data_dir)

        self.data_files = [join(self.data_dir, f) for f in self.files]
        self.label_files = [join(self.labels_dir, f) for f in self.files]

    def __iter__(self):
        n_train = self.n_train()

        if self.mode == 'train':
            current = 0
            endId = n_train
        elif self.mode == 'test':
            current = n_train
            endId = len(self.data_files)

        while current < endId:
            # todo: load images and labels
            # hint: scale images between 0 and 1
            # hint: if training takes too long or memory overflow, reduce image size!
            resized_size = 572
            # load images
            data_image = Image.open(self.data_files[current])
            data_image = data_image.resize((resized_size, resized_size))

            # load labels
            label_image = Image.open(self.label_files[current])
            label_image = label_image.resize((resized_size, resized_size))

            # apply data augmentation and normalization
            data_image, label_image = self.__applyDataAugmentation(data_image, label_image)
            current += 1
            yield (data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))
    
    def __applyDataAugmentation(self, data_image, label_image):
        
        flipOption = random.randint(0,2)
        
        zoomOption = random.randint(0,2)
                   
        rotateOption = random.randint(0,2)
                   
        gammaOption = random.randint(0,1)
         
        elasticOption = random.randint(0,1)

        data_image = self.__flip(data_image, flipOption)
        label_image = self.__flip(label_image, flipOption)
        
        data_image = self.__zoom(data_image, zoomOption)
        label_image = self.__zoom(label_image, zoomOption)
        
        data_image = self.__rotate(data_image, rotateOption)
        label_image = self.__rotate(label_image, rotateOption)

        #data_image = self.__gamma(data_image, gammaOption)
        
        # normalization
        data_image = np.asarray(data_image, dtype=np.float32) / 255.
        label_image = np.asarray(label_image, dtype=np.float32)
        
        return data_image, label_image
    def __flip(self, image, flipOption):
        if flipOption == 1:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        elif flipOption == 2:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        return image

    def __zoom(self, image, zoomOption):
        resized_size, _ = image.size
        crop_ratio = 1
        if zoomOption == 1:
            crop_ratio = 0.95
        elif zoomOption == 2:
            crop_ratio = 0.9
        crop_start = int(resized_size*(1-crop_ratio)/2)
        crop_size = int(resized_size*crop_ratio)
        crop_pos= (crop_start,crop_start,crop_start+crop_size,crop_start+crop_size)
        image = image.crop(crop_pos).resize((resized_size, resized_size))
        return image

    def __rotate(self, image, rotateOption):
        if rotateOption == 1:
            image = image.transpose(Image.ROTATE_90)
        elif rotateOption == 2:
            image = image.transpose(Image.ROTATE_180)
        return image

    def __gamma(self, image, gammaOption):
        gamma = 1
        if gammaOption == 1:
            gamma = 0.8
        image = transforms.functional.adjust_gamma(image, gamma, gain=1)
        return image

  
