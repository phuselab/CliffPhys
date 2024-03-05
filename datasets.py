#================================================================================================================================
#==            'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#================================================================================================================================

"""
This script contains classes for handling the BP4D+ dataset and its XYZ version (BP4D_XYZ),
and the DatasetBase class, which allows the definition of any new dataset.

DATASETS:
    This script defines classes to handle video and ground truth datasets. Each class includes methods for loading videos and 
    their corresponding ground truth data. The BP4D class also provides a function to create the XYZ version of the 
    dataset by calling the utils.get_XYZ_tensor function.
    Conversely, the BP4D_XYZ dataset class provides only a method for loading each XYZ tensor video alongside 
    its corresponding ground truth.
"""

import os
import glob
import numpy as np
import utils
#import errors
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt

class DatasetBase:
    """
    Base class for datasets.

    Attributes:
        data_dir (str): Directory where the dataset is stored.
    """
    def __init__(self):
        self.data_dir = './data/'

    def load_dataset(self):
        raise NotImplementedError("Subclasses must implement load_dataset method")

class BP4D(DatasetBase):
    """
    Class for the BP4D+ dataset (In this demo, original videos are not provided. 
    Instead, sample videos converted into XYZ tensors will be provided in the ./data directory).

    Attributes:
        name (str): Name of the dataset.
        path (str): Path to the dataset directory.
        fs_gt (int): Ground truth sampling frequency.
        data (list): List to store each produced XYZ video tensor.
    """
    def __init__(self):
        super().__init__()
        self.name = 'bp4d'
        self.path = self.data_dir + self.name + '/'
        self.fs_gt = 1000
        self.data = [] 

    def load_dataset(self):

        print('\nLoading dataset ' + self.name + '...')
        for sub in utils.sort_nicely(os.listdir(self.path)):
            sub_path = self.path + sub + '/'
            video_path = sub_path + 'vid.avi'

            if os.path.exists(video_path):
                d = {}
                d['video_path'] = video_path
                d['subject'] = sub
                d['xyz_tensor'] = []
                d['gt'] = self.load_gt(sub_path)
                self.data.append(d)

        print('%d items loaded!' % len(self.data))

    def load_gt(self, sub_path):
        #Load GT
        gt = np.loadtxt(sub_path + "/Resp_Volts.txt")
        return gt
	
    def extract_XYZ(self, video_path, fps, data_type, extraction_OF, extraction_depth, paths, gt=None, fs_gt=None):
        if data_type == 'xyz_tensor':
            if not all(bool(glob.glob(os.path.join(path, "*.npy"))) for path in paths.values()):
                utils.get_XYZ_tensor(video_path, fps, extraction_OF, extraction_depth, paths, gt, fs_gt)
            else:
                print("> File XYZ tensor already exists!")

class BP4D_XYZ(DatasetBase):
    """
    Class for the BP4D_XYZ dataset.

    Attributes:
        name (str): Name of the dataset.
        path (str): Path to the dataset directory.
        fs_gt (int): Ground truth sampling frequency.
        data (list): List to store each XYZ video tensor.
    """
    def __init__(self, generation_method):
        super().__init__()
        self.name = 'bp4d_XYZ'
        self.path = self.data_dir + self.name + '/'
        self.generation_method = generation_method
        self.fs_gt = 1000
        self.fps = 25
        self.new_fps = 20
        self.data = [] 

    def load_dataset(self):

        print('\nLoading dataset ' + self.name + '...')
        for sub in utils.sort_nicely(os.listdir(self.path)):
            sub_path = self.path + sub + '/'

            if os.path.exists(sub_path):
                d = {}
                d['name'] = self.name
                d['video_path'] = sub_path
                d['subject'] = sub
                d['xyz_tensor'] = []
                d['gt'] = self.load_XYZ_gt(sub)
                self.data.append(d)

        print('%d items loaded!' % len(self.data))
	
    def load_XYZ_gt(self, sub):
        if os.path.exists(os.path.join(self.path, sub, 'gt.npy')):
            return np.load(os.path.join(self.path, sub, 'gt.npy'))
        else:
            auxiliary_dataset = BP4D()
            return auxiliary_dataset.load_gt(os.path.join(auxiliary_dataset.path, sub))
