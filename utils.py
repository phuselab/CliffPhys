#================================================================================================================================
#==            'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#================================================================================================================================

"""
Code containing utility functions for processing respiratory data and videos.

UTILS:
    This script provides the following functions:
    - sort_nicely: Sorts the list of files in a human-readable way.
    - get_vid_stats: Retrieves the duration and FPS of a video.
    - get_vid_stats_tensor: Retrieves the FPS of a XYZ video.
    - extract_frames_yield: Yields frames from a video file.
    - video_preprocessing_testing: Prepares video data for testing/prediction phase.
    - get_XYZ_tensor: Extracts XYZ motion tensors from a video.
    - filter_RW: Applies band-pass filtering to a signal (cutoffs frequencies 0.1 Hz, 0.5 Hz).
    - sliding_straded_win_idx: Computes indices for overlapping or non-overlapping windows.
    - sig_windowing: Performs signal windowing.
    - tensor_gt_windowing: Slices a XYZ tensor and its ground truth into windows.
    - Welch_rpm: Computes the Welch periodogram of a respiratory signal.
    - sig_to_RPM: Converts a signal to respiratory rate in RPM.
    - print_signal_psd: Prints the Power Spectral Density of a signal.

    The get_XYZ_tensor employs the MiDaS family of models, specifically the DPT-Large model with SwinV2 
    encoder backbon to perform the Monocular Depth Estimation phase
    The get_XYZ_tensor employs the Raft-Small model, to compute the Optical Flow x-axis and y-axis projections.

"""

from __future__ import division
import torch
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
import cv2 as cv
import re 
import os
import warnings
import importlib
warnings.filterwarnings("ignore") 

def sort_nicely(l): 
    """ Sort the given list in the way that humans expect. 
    """ 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    l.sort( key=alphanum_key ) 
    return l

def get_vid_stats(videoFileName):
    """
    Retrieves the duration and FPS of a video.
    """
    cap = cv.VideoCapture(videoFileName)
    fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    return duration, int(fps)

def get_vid_stats_tensor(videoFileName):
    """
    Retrieves the FPS of a XYZ video.
    """
    fps = np.load(os.path.join(videoFileName, 'fps.npy'))
    return None, int(fps)

def extract_frames_yield(videoFileName):
    """
    This method yield the frames of a video file name or path.
    """
    vidcap = cv.VideoCapture(videoFileName)
    success, image = vidcap.read()
    while success:
        yield image
        success, image = vidcap.read()
    vidcap.release()

def video_preprocessing_testing(data, F):
    """
    Prepares video data for testing/prediction phase.
    """
    data = data[np.newaxis, :F, ...]
    return data

def get_XYZ_tensor(video_path, fps, extraction_OF, extraction_depth, paths, gt=None, fs_gt=None):
    """
    Extracts XYZ motion tensors from a video. This methods employs the MiDaS family of models, 
    specifically the DPT-Large model with SwinV2 encoder backbon to perform the Monocular Depth Estimation phase.
    It also uses the Raft-Small method to compute the vertical and horizontal Optical Flow components.
    """

    ptlflow = importlib.import_module('ptlflow')
    #from ptlflow import models_dict
    print("\nExtracting XYZ motion tensor...")

    #_, fps = get_vid_stats(video_path)

    frames_X = []
    frames_Y = []
    frames_Z = []

    if extraction_depth == 'midas':
        #from depth.depth_midas import DepthSignalProcessing
        depth_midas = importlib.import_module('depth.depth_midas')
        model_type="dpt_swin2_large_384"
        depth_extractor = depth_midas.DepthSignalProcessing(model_type=model_type)
    
    if extraction_OF in ptlflow.models_dict:
        #import ptlflow
        #from ptlflow.utils import flow_utils
        #from ptlflow.utils.io_adapter import IOAdapter
        model_of = extraction_OF
        ckpt = 'things'
        batch_size = 64
        cuda=True
        if not cuda:
            torch.cuda.is_available = lambda : False
            device = 'cpu'
        else:
            device = torch.device("cuda")
        OFmodel = ptlflow.get_model(model_of, pretrained_ckpt=ckpt)
        OFmodel.to(device)
    elif extraction_OF == 'OF_model':
        #from motion.motion import OF
        motion = importlib.import_module('motion.motion')
        model_of = motion.OF
    
    frames_PIL = [Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB)) for frame in extract_frames_yield(video_path)]

    newsize = (256, 256)
    frames_PIL_res = [frame.resize(newsize) for frame in frames_PIL]
    frames_res = [np.array(frame) for frame in frames_PIL_res]
    nframes = len(frames_res)

    print("\n> Computing Optical Flow...")
    exceeding = None

    while True:
        try:
            print("\n> Attempting with batch size: " + str(batch_size))
            for i in tqdm(range(0, nframes, batch_size)):
                if i == 0:
                    start = i
                else:
                    start = i-1
                end = min(i+batch_size, nframes)
                batch = frames_res[start:end]
                if len(batch) <= 2:
                    exceeding = len(batch)
                    batch = frames_res[start-len(batch):end]
                if i == 0:
                    io_adapter = ptlflow.utils.io_adapter.IOAdapter(OFmodel, batch[0].shape[:2], cuda=cuda)
                inputs = io_adapter.prepare_inputs(batch)
                input_images = inputs["images"][0]
                video1 = input_images[:-1]
                video2 = input_images[1:]
                input_images = torch.stack((video1, video2), dim=1)
                if cuda:
                    input_images = input_images.cuda()
                inputs["images"] = input_images
                predictions = OFmodel(inputs)
                predictions = io_adapter.unpad_and_unscale(predictions)
                flows_horiz = torch.squeeze(predictions['flows'])[:,0,:,:].cpu().detach().numpy() 
                flows_vert = torch.squeeze(predictions['flows'])[:,1,:,:].cpu().detach().numpy()    
                if exceeding is not None :
                    flows_horiz = flows_horiz[exceeding:]
                    flows_vert = flows_vert[exceeding:]
                    exceeding == None
                frames_X.append(flows_horiz)
                frames_Y.append(flows_vert)
            break
        except RuntimeError:
            batch_size = batch_size // 2
            frames_X = []
            frames_Y = []
            if batch_size < 4:
                raise ValueError("Batch size is too tiny, maybe need more GPU memory.")
    
    del OFmodel
    torch.cuda.empty_cache()

    print("\n> Computing Depth estimate...")
    
    for i, frame in tqdm(enumerate(frames_PIL_res)):
        depth_extractor.extract_depth(frame)
        frames_Z.append(depth_extractor.get_frame_depth())
        depth_extractor.set_frame_depth(None)       
        if extraction_depth == 'midas':
            depth_extractor.set_first_execution(True)
    
    print("\n> Creating XYZ tensor...")

    frames_XX = [cv.cvtColor(frame, cv.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame for frame in np.concatenate(frames_X)]
    frames_YY = [cv.cvtColor(frame, cv.COLOR_RGB2GRAY) if len(frame.shape) == 3 else frame for frame in np.concatenate(frames_Y)]
    frames_ZZ = [cv.cvtColor(np.array(frame), cv.COLOR_RGB2GRAY) if len(np.array(frame).shape) == 3 else np.array(frame) for frame in frames_Z[:-1]]
    frames_res = np.array(frames_res[:-1])
    
    # Channels: 0 : R, 1 : G, 2 : B, 3 : X, 4 : Y, 5 : Z
    frames_tensor = np.stack([np.array(frames_res)[..., 0], np.array(frames_res)[..., 1], 
                              np.array(frames_res)[..., 2], np.array(frames_XX), np.array(frames_YY),                         
                              np.array(frames_ZZ)], axis=-1)
    
    np.save(paths['XYZ']+'/RGB_XYZ_tensor', frames_tensor)
    np.save(paths['XYZ']+'/fps', fps)
    if gt is not None and fs_gt is not None:
        np.save(paths['gt']+'/gt', gt)
        np.save(paths['gt']+'/fs_gt', fs_gt)

def filter_RW(sig, fps, lo=0.1, hi=0.5):
    """
    This method performs posptprocessing steps of fiedler methods; the postprocessing process performs on the signal a normalization, computes the gradient of the signal and applies a band-pass filter

    Parameters
    ----------
        sig: the considered signal
        fps : the fps of the considered video

    Returns
    -------
        the postprocessed signal
    """
    #sig = np.diff(np.asarray(sig), axis=0)
    #sig = np.squeeze(sig)
    if (sig.ndim == 1):
        sig = sig[np.newaxis,:]

    b, a = signal.butter(N=2, Wn=[lo, hi], fs=fps, btype='bandpass')
    filtered_sig = signal.filtfilt(b, a, sig)

    return filtered_sig

def sliding_straded_win_idx(N, wsize, stride, fps):
    """
    This method is used to compute the indices for creating an overlapping windows signal.

    Args:
        N (int): length of the signal.
        wsize (float): window size in seconds.
        stride (float): stride between overlapping windows in seconds.
        fps (float): frames per seconds.

    Returns:
        List of ranges, each one contains the indices of a window, and a 1D ndarray of times in seconds, where each one is the center of a window.
    """
    wsize_fr = wsize*fps
    stride_fr = stride*fps
    idx = []
    timesES = []
    num_win = int((N-wsize_fr)/stride_fr)+1
    s = 0
    for i in range(num_win):
        idx.append(np.arange(s, s+wsize_fr))
        s += stride_fr
        timesES.append(wsize/2+stride*i)
    return idx, np.array(timesES, dtype=np.float32)

def sig_windowing(sig, fps, wsize, stride=1):
    """ Performs signal windowing

    Args:
      sig (list/array): full signal
      fps       (float): frames per seconds      
      wsize     (float): size of the window (in seconds)
      stride    (float): stride (in seconds)

    Returns:
      win_sig (list): windowed signal
      timesES (list): times of (centers) windows 
    """
    sig = np.array(sig).squeeze()
    block_idx, timesES = sliding_straded_win_idx(sig.shape[0], wsize, stride, fps)
    sig_win  = []
    for e in block_idx:
        st_frame = int(e[0])
        end_frame = int(e[-1])
        wind_signal = np.copy(sig[st_frame: end_frame+1])
        sig_win.append(wind_signal[np.newaxis, :])

    return sig_win, timesES

def tensor_windowing(tensor, window_size):
        windows_tensor = []
        num_windows = tensor.shape[0] // window_size

        for i in range(num_windows):
            start_index = i * window_size
            end_index = start_index + window_size
            window_tensor = tensor[start_index:end_index, :, :, :]
            windows_tensor.append(window_tensor)
        
        return windows_tensor

def Welch_rpm(resp, fps, winsize, fRes=0.1):
    """
    This method computes the spectrum of a respiratory signal

    Parameters
    ----------
        resp: the respiratory signal
        fps: the fps of the video from which signal is estimated
        winsize: the window size used to compute spectrum
        minHz: the lower bound for accepted frequencies
        maxHz: the upper bound for accepted frequencies

    Returns
    -------
        the array of frequencies and the corrisponding PSD
    """
    step = 1
    nperseg=fps*winsize
    noverlap=fps*(winsize-step)

    nyquistF = fps/2
    nfft = max(2048, (60*2*nyquistF) / fRes)

    # -- periodogram by Welch
    F, P = signal.welch(resp, nperseg=nperseg, noverlap=noverlap, fs=fps, nfft=nfft)
    F = F.astype(np.float32)
    P = P.astype(np.float32)

    Pfreqs = F*60
    Power = P
    return Pfreqs, Power

def sig_to_RPM(sig, fps, winsize):
    sig = np.vstack(sig)

    Pfreqs, Power = Welch_rpm(sig, fps, winsize)
    Pmax = np.argmax(Power, axis=1)  # power max
    rpm = Pfreqs[Pmax.squeeze()]

    if (rpm.size == 1):
        return rpm.reshape(1, -1)

    return rpm

def print_signal_psd(name, sig, gt, winsize, fps, results_dir):

    figs_path = './'+results_dir+'/Figs/'
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)
    

    sig = (sig[0, :] - np.mean(sig[0, :]))/np.std(sig[0, :])
    gt = (gt[0, :] - np.mean(gt[0, :]))/np.std(gt[0, :])

    gt = signal.resample(gt, sig.shape[0])

    gt = gt[np.newaxis,:]
    sig = sig[np.newaxis,:]

    Pfreqs_gt, Power_gt = Welch_rpm(gt, fps, winsize)
    Pfreqs_sig, Power_sig = Welch_rpm(sig, fps, winsize)

    import seaborn as sns
    sns.set_context('poster')
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    time_x = np.linspace(0, gt.shape[1]/fps, gt.shape[1])
    axs[0].plot(time_x, sig[0, :], label='Filtered Prediction')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Prediction filtered [0.1, 0.5] Hz')
    axs[1].plot(time_x, gt[0, :])
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Ground Truth')
    axs[1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(figs_path + "signals_sbj_" + name + ".pdf", bbox_inches='tight', pad_inches=0, format='pdf')

    plt.figure(figsize=(10, 6))
    plt.plot(Pfreqs_gt[0:1200], Power_gt[0, 0:1200], label='Ground Truth')
    plt.plot(Pfreqs_sig[0:1200], Power_sig[0, 0:1200], label='Prediction')
    plt.legend()
    plt.title('Welch Periodogram ')
    plt.xlabel('RPM')
    plt.ylabel('Power Spectral Density')
    plt.tight_layout()
    plt.savefig(figs_path + "psd_sbj_" + name + ".pdf", bbox_inches='tight', pad_inches=0, format='pdf')
