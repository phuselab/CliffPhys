#=================================================================================================================================
#==             'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#=================================================================================================================================

"""
Code for the preprocessing and postprocessing phases of the 'CliffPhys' family of respiratory waveform prediction methods.
(the MethodBase allows to define any new prediction method)

METHODS:
    This script defines the MethodBase class and the CliffPhys class for processing respiratory waveform extraction from videos using Clifford algebra-based deep extraction.

    MethodBase:
        - Generic Base class for any respiratory waveform estimating methods.
        - Contains attributes and a method that must be implemented by subclasses.

    CliffPhys:
        - Class for extracting respiratory waveform from videos using the CliffPhys family of models.
        - Contains attributes for model configuration and preprocessing.
        - Implements methods for data preprocessing and waveform prediction.

"""

import os
import numpy as np
from scipy import signal
from scipy import ndimage
import cv2 as cv
import utils
from tqdm import tqdm
import importlib

class MethodBase:
    def __init__(self):
        """
        Base class for processing methods.

        Attributes:
            name (str): The name of the method.
        """
        self.name = ''

    def process(self, data):
        """
        Process method. Must be implemented by subclasses.

        Parameters:
            data (dict): Input data.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Subclasses must implement process method")

class CliffPhys(MethodBase):
    """
    CliffPhys class for extracting respiratory waveform from videos using Clifford algebra-based deep extraction.

    Attributes:
        model_name (str): The name of the model.
        version (str): The version of the model ('CliffPhys30_d' is the top performing one).
        F (int): Number of input frames per window fed to the network (set to 399, almost 20 seconds at 20 fps ).
        H (int): Height (or width) of each frame (in pixels).
        new_fps (int): Sampling frequency at which the input tensor is resampled.
        module_name (str): Name of the module to be imported, which contains the 'CliffPhys' model.
        model_params (dict): Model parameters.
        data_means (numpy.ndarray): Means of the three data channels.
        data_stds (numpy.ndarray): Standard deviations of the three data channels.
        label_mean (numpy.ndarray): Mean of the GTs.
        label_std (numpy.ndarray): Standard deviation of the GTs.
    """

    def __init__(self, model, version):
        super().__init__()
        self.model_name = model
        self.version = version
        self.name = 'cliffphys'
        self.module_name = 'clifford'
        self.F = 399
        self.H = 36
        self.new_fps = 20
        self.model_params = {'img_size':self.H, 'num_frames': self.F}
        self.data_means = np.load('./data/cohface_test_means_stds/data_means.npy')
        self.data_stds = np.load('./data/cohface_test_means_stds/data_stds.npy')
        self.label_mean = np.load('./data/cohface_test_means_stds/label_means.npy')
        self.label_std = np.load('./data/cohface_test_means_stds/label_stds.npy')

    def process(self, data):
        """
        Process method for CliffPhys.

        Parameters:
            data (dict): Input data.

        Returns:
            numpy.ndarray: Predicted respiratory waveform.
        """
        torch = importlib.import_module('torch')
        module = importlib.import_module(self.module_name)
        model_class = getattr(module, self.model_name)
        processor_class = getattr(module, 'Processor')

        model_dir = os.path.join('.', 'weights', self.model_name, self.version, '')
        x = self.preprocess_data(data)
        
        preds = []
        for i in tqdm(range(len(x)), desc="Predicting"):
            tqdm.write("> Predicting waveform for window %s, using %s model" % (str(i), self.model_name))
            x_ = utils.video_preprocessing_testing(x[i], self.F)

            tester = processor_class(model_class, self.model_params, load_path=model_dir+'/')

            test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(x_))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
            prediction = tester.predict(test_loader)
            preds.append(prediction)

        preds = np.array(preds).squeeze()

        tqdm.write("> Stacking prediction windows...")
        preds = preds.reshape((preds.shape[0]*preds.shape[1],)) if preds.ndim > 1 else preds.reshape((preds.shape[0],))
        return np.stack(preds)
    
	
    def preprocess_data(self, data):
        """
        Preprocess input data for CliffPhys.

        Parameters:
            data (dict): Input data.

        Returns:
            numpy.ndarray: Preprocessed tensor data.
        """
        tensor = data['xyz_tensor'][..., 3:]
           
        #Video tensor spatial resampling
        tensor = np.array([cv.resize(frame, (36, 36)) for frame in tensor])

        #Video tensor temporal resampling
        tensor = ndimage.zoom(tensor, (self.new_fps/data['fps'], 1, 1, 1), order=1)

        #Windowing both GT and video tensor
        tensor_w = utils.tensor_windowing(tensor, self.F)

        #GT and video tensor standardization
        for w in range(len(tensor_w)):
            for channel in range(tensor_w[w].shape[-1]):
                mean_d = float(self.data_means[0, (channel+1)%3])
                std_d = float(self.data_stds[0, (channel+1)%3])
                tensor_w[w][..., channel] = (tensor_w[w][..., channel] - mean_d) / std_d

        tensor_w = np.array(tensor_w)

        # Z -> scalar (position 0), X -> vector component (position 1), Y -> vector component (position 2)
        tensor_w = tensor_w[..., [2, 0, 1]]
            
        if tensor_w.ndim == 4:
            tensor_w = tensor_w.transpose((0, 3, 1, 2))
        elif tensor_w.ndim == 5:
            tensor_w = tensor_w.transpose((0, 1, 4, 2, 3))

        #Remove the Z component if the model does not expect it as additional information 
        #(i.e., if the model name does not have an extension '_d').
        if '_d' not in self.model_name:
            tensor_w = tensor_w[:, :, 1:, :, :]
        
        return tensor_w
