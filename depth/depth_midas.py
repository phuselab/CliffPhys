import os
import glob
import torch
import MiDaS.utils
import cv2
import argparse
import time

import numpy as np
from PIL import Image

from midas.model_loader import default_models, load_model

class DepthSignalProcessing:
    def __init__(self, model_weights=None, model_type="dpt_beit_large_512", optimize=False, side=False, grayscale=True):
        self.model_weights = "./MiDaS/weights/"+model_type+".pt" #default_models[model_type] if model_weights is None else model_weights
        self.model_type = model_type
        self.optimize = optimize
        self.side = side
        self.grayscale = grayscale
        self.use_camera = False

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.model, self.transform, self.net_w, self.net_h = load_model(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
            self.model_weights, 
            self.model_type, 
            self.optimize, 
            None,  # height 
            False  # square 
        )

        self.first_execution = True
        self.frame_depth = None
    
    def process(self, device, image, target_size):
        """
        Run the inference and interpolate.

        Args:
            device (torch.device): the torch device used
            model: the model used for inference
            model_type: the type of the model
            image: the image fed into the neural network
            input_size: the size (width, height) of the neural network input (for OpenVINO)
            target_size: the size (width, height) the neural network output is interpolated to
            optimize: optimize the model to half-floats on CUDA?
            use_camera: is the camera used?

        Returns:
            the prediction
        """

        input_size = (self.net_w, self.net_h)

        if "openvino" in self.model_type:
            if self.first_execution or not self.use_camera:
                #print(f"    Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")
                self.first_execution = False

            sample = [np.reshape(image, (1, 3, *input_size))]
            prediction = self.model(sample)[self.model.output(0)][0]
            prediction = cv2.resize(prediction, dsize=target_size,
                                    interpolation=cv2.INTER_CUBIC)
        else:
            sample = torch.from_numpy(image).to(device).unsqueeze(0)

            if self.optimize and device == torch.device("cuda"):
                if self.first_execution:
                    print("  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                        "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                        "  half-floats.")
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            if self.first_execution or not self.use_camera:
                height, width = sample.shape[2:]
                #print(f"    Input resized to {width}x{height} before entering the encoder")
                self.first_execution = False

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=target_size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return prediction


    def extract_depth(self, image: Image.Image) -> Image.Image:
        # Convert PIL Image to numpy array and normalize
        original_image_rgb = np.array(image) / 255.0  # in [0, 1]
        
        image = self.transform({"image": original_image_rgb})["image"]

        with torch.no_grad():
            prediction = self.process(
                torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
                image, 
                original_image_rgb.shape[1::-1]
            )

        self.frame_depth = Image.fromarray(prediction)


    def get_frame_depth(self):
        return self.frame_depth
    
    def set_frame_depth(self, frame_depth):
        self.frame_depth = frame_depth

    def get_first_execution(self):
        return self.first_execution

    def set_first_execution(self, first_execution):
        self.first_execution = first_execution
