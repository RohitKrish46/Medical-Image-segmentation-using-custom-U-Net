import os
from os.path import isdir, exists, abspath, join
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image

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
            
            #resize dimension
            resize_row = 572
            resize_col = 572
            # load images
            data_image = Image.open(self.data_files[current])
            
            # load labels
            label_image = Image.open(self.label_files[current])
            
            # applying data augmentation
            data_image, label_image = self.Augmentation(data_image, label_image)
            
             # normalization
            data_image = data_image.resize((resize_row, resize_col))
            label_image = label_image.resize((resize_row, resize_col))
            data_image = np.asarray(data_image, dtype=np.float32) / 255.
            label_image = np.asarray(label_image, dtype=np.float32)
           
            current += 1
            yield (data_image, label_image)
            
    def Augmentation(self, data_image, label_image):
        #flipping
        flip = random.randint(0,2)
        if flip == 1:
            data_image = transforms.functional.hflip(data_image)
            label_image = transforms.functional.hflip(label_image)
            
        elif flip == 2:
            data_image = transforms.functional.vflip(data_image)
            label_image = transforms.functional.vflip(label_image)
        #rotation                
        rotate = random.randint(0,2)
        if rotate == 1:
            data_image = data_image.transpose(Image.ROTATE_90)
            label_image = label_image.transpose(Image.ROTATE_90)
        elif rotate == 2:
            data_image = data_image.transpose(Image.ROTATE_180)
            label_image = label_image.transpose(Image.ROTATE_180)
        #gamma correction           
        gamma_val = random.randint(0,1)
        gamma = 1
        if gamma_val == 1:
            gamma = 0.8
        data_image = transforms.functional.adjust_gamma(data_image, gamma, gain=1)
                       
        return data_image, label_image
 
    def setMode(self, mode):
        self.mode = mode

    def n_train(self):
        data_length = len(self.data_files)
        return np.int_(data_length - np.floor(data_length * self.test_percent))

  
