import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from torch.utils.data import Dataset, DataLoader

import numpy as np
from pathlib import Path
import os
import glob
from os import listdir
from os.path import isfile, join

from PIL import Image
import PIL

from skimage import io, color
#import splitfolders

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ColorizationDataset(Dataset):
    """CIFAR 10 dataset."""

    def __init__(self, path_list, transform=None):
        """
        Args:
            list of paths (string): Path to the png file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_list = path_list
        self.transform = transform

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        
        img = io.imread(self.path_list[idx])
        
        img = img.astype('float32')
        
        
        
        new_image = color.rgb2xyz(img)
        new_image = color.xyz2lab(new_image)
        
        img = img.transpose((2, 1, 0))
        #img = img.transpose((2, 0, 1))
        new_image = new_image.transpose((2, 1, 0))
        
        
        L = new_image[0, :, :]
        ab = new_image[1:3, :, :]
        
        #L = torch.tensor(np.expand_dims(L, axis=0))
        
        L = np.expand_dims(L, axis=0)
        
       
        
        sample = {'L': L, 'ab': ab, 'image': img}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        L, ab, imgage = sample['L'], sample['ab'], sample['image']
         
        # numpy image: H x W x C
        # torch image: C X H X W
        
        
        
        return {'L': torch.from_numpy(L),
                'ab': torch.from_numpy(ab),
                'image': torch.from_numpy(imgage)}
    
    
class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""        

    def __call__(self, sample):
        
        L, ab, imgage = sample['L'], sample['ab'], sample['image']
        L_copy = np.copy(L)
        ab_copy = np.copy(ab)
        imgage_copy = np.copy(imgage)

        # convert image to grayscale
        #image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # scale color range from [0, 255] to [-1, 1]
        
        norm_imgage_copy = (imgage_copy - np.min(imgage_copy)) / (np.max(imgage_copy) - np.min(imgage_copy))
        norm_imgage_copy = 2*norm_imgage_copy-1
        
        norm_L_copy = (L_copy - np.min(L_copy)) / (np.max(L_copy) - np.min(L_copy))
        norm_L_copy = 2*norm_L_copy-1
        
        norm_ab_copy = (ab_copy - np.min(ab_copy)) / (np.max(ab_copy) - np.min(ab_copy))
        norm_ab_copy = 2*norm_ab_copy-1
        
        
        norm_imgage_copy[np.isnan(norm_imgage_copy)] = 0
        norm_L_copy[np.isnan(norm_L_copy)] = 0
        norm_ab_copy[np.isnan(norm_ab_copy)] = 0
        #imgage_copy=  imgage_copy/255.0
        #L_copy=  L_copy/255.0
        #ab_copy=  ab_copy/255.0
        
        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        #key_pts_copy = (key_pts_copy - 100)/50.0
        
        

        return {'L': norm_L_copy,
                'ab': norm_ab_copy,
                'image': norm_imgage_copy}

