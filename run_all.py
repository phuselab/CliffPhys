#=================================================================================================================================
#==             'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)              ==
#=================================================================================================================================

"""
Demo code for the submitted paper 'CliffPhys: Camera-based Respiratory Measurement using Clifford Neural Networks' (Paper ID #11393)

MAIN:
    This demo script facilitates the extraction of respiratory waveforms from videos using the CliffPhys family of methods.

    Action 0:   Extract respiratory waveforms from provided videos.

        extract_respiration:
        - Function for extracting respiratory waveforms from videos.
        - Parameters: dataset (object), methods (list), video_type (str), results_dir (str).
        - Calls utils functions for data loading and preprocessing.

                DEMO: python run_all.py -a 0 -d './estimates/' -t 'XYZ' -m 'CliffPhys03_d'

    Action 1:   Extract Respiratory Per Minute (RPM) from predicted and ground truth waveforms.

        extract_rpms:
        - Function for extracting Respiratory Per Minute (RPM) from predicted and ground truth waveforms.
        - Parameters: results_dir (str), visualize (bool).
        - Calls utils functions for data processing and visualization.

                DEMO: python run_all.py -a 1 -d './estimates/'

    Action 2:   Print computed error metrics.

        print_metrics:
        - Function for printing computed error metrics from RPMs.
        - Parameters: results_dir (str).
        - Calls utils functions for data loading and error computation.

                DEMO: python run_all.py -a 2 -d './estimates/'

"""

import os
import numpy as np
import pickle
import utils
import datasets
from tqdm import tqdm
import sys, getopt
from prettytable import PrettyTable
from errors import RMSEerror, MAEerror, MAPEerror, PearsonCorr, LinCorr
from methods import CliffPhys

def extract_respiration(dataset, methods, video_type, results_dir): 
    """
    Function that extracts the respiratory wavefor from videos.

    Parameters:
        dataset (object): Dataset object.
        methods (list): List of methods.
        video_type (str): Type of video ('rgb' of 'xyz').
        results_dir (str): Directory to save results.
    """
	
    dataset.load_dataset()
    video_type = video_type.lower()

    # For each dataset subject...
    for d in tqdm(dataset.data, desc="Processing files"):
        outfilename = results_dir + dataset.name + '_' + d['subject'] + '.pkl'
	
        if os.path.exists(outfilename):
            file = open(outfilename, 'rb')
            data = pickle.load(file)
            if isinstance(data['estimates'][0]['estimate'], np.ndarray):
                tqdm.write("> File %s already exists! Skipping..." % outfilename)
                continue
        
        d['fs_gt'] = dataset.fs_gt
		
        results = {'video_path': d['video_path'],
					'gt' : d['gt'],
					'fs_gt': d['fs_gt'],
					'estimates': [] }
		
        tqdm.write("> Processing video %s\n" % d['subject'])
		
        # For each estimating method...
        for m in methods:
            tqdm.write("> Applying method %s ..." % m.name)
			
            # If video_type is 'rgb', extracts the XYZ video tensor
            if video_type == 'rgb':
                paths = {
					'XYZ': os.path.join(dataset.data_dir, dataset.name+'_XYZ', 'XYZ', m.extract_OF_type+'-'+m.extract_depth_type, d['subject']),
					'gt': os.path.join(dataset.data_dir, dataset.name+'_XYZ', 'gt', d['subject'])
				}
				
                _, d['fps'] = utils.get_vid_stats(d['video_path'])
				
                for key, path in paths.items():
                    if not os.path.exists(path):
                        os.makedirs(path)
				
                dataset.extract_XYZ(d['video_path'], d['fps'], m.data_type, m.extract_OF_type, m.extract_depth_type, paths, d['gt'], d['fs_gt'])
                tensor_file = os.path.join(paths['XYZ'], 'RGB_XYZ_tensor.npy')

            # If video_type is 'xyz', load the XYZ video tensor									
            elif video_type == 'xyz':
                _, d['fps'] = utils.get_vid_stats_tensor(d['video_path'])
                results['fps'] = 20

                tensor_file = os.path.join(d['video_path'], 'RGB_XYZ_tensor.npy')

                if not os.path.exists(tensor_file):
                    tqdm.write("> XYZ data for dataset %s has not been generated yet. Skipping method..." % dataset.name)
                    break
				
            else:
                tqdm.write("> Unsupported video type %s already exists! Exiting..." % video_type)
                continue

            # Estimate the respiratory wavefor from the XYZ video tensor, using method m
            d['xyz_tensor'] = np.load(tensor_file)
			
            output = {'method': m.name,
					'estimate': m.process(d)}

            results['estimates'].append(output)
		
        d['xyz_tensor'] = []
		
        with open(outfilename, 'wb') as fp:
            pickle.dump(results, fp)
            tqdm.write('> Results saved!\n')

def extract_rpms(results_dir, visualize=False):
    """
    Function that extracts RPMs (Respiratory Per Minute) from predicted and GT respiratory waveforms, save them in a rpms.pkl file.

    Parameters:
        results_dir (str): Directory containing both predicted and GT respiratory waveforms.
        visualize (bool): Flag to visualize respiratory signals and Power Spectral Densities estimations (PSDs).
    """
    print('\n> Loading extracted data from ' + results_dir + '...')

    method_rpms = {}

    files = utils.sort_nicely(os.listdir(results_dir))

    # For each dataset subject...
    for filepath in tqdm(files, desc="Processing files"):
        tqdm.write("> Processing file %s" % (filepath))

        if filepath.endswith('.pkl'):

            if 'metrics' in filepath:
                continue

            # Open the file with pickled data
            file = open(results_dir + filepath, 'rb')
            data = pickle.load(file)
            file.close()

            # Extract ground truth data
            fs_gt = data['fs_gt']
            gt = data['gt']

            # Filter ground truth (ws is the whole signal duration)
            filt_gt = utils.filter_RW(gt, fs_gt)
            ws = filt_gt.shape[1] / fs_gt

            tqdm.write("> Length: %.2f sec" % (len(gt) / int(fs_gt)))

            # Extract ground truth RPM using Welch with (win_size/1.5)
            gt_rpm = utils.sig_to_RPM(filt_gt, fs_gt, int(ws/1.5)) 

            # Extract estimation data
            fps = data['fps']

            # For each estimation method...
            for i, est in enumerate(data['estimates']):

                # Extract the method name
                cur_method = est['method']

                # Extract predicted data
                sig = np.squeeze(est['estimate'])

                # Filter predicted data (ws is the whole signal duration)
                filt_sig = utils.filter_RW(sig, fps)
                ws = len(sig) / fps
                
                # visualize signals and PSDs
                if visualize:
                    utils.print_signal_psd(filepath, filt_sig, filt_gt, int(ws/1.5), fps, results_dir)

                # Extract estimated RPM using Welch with (win_size/1.5)
                sig_rpm = utils.sig_to_RPM(filt_sig, fps, int(ws/1.5))

                rpms = [sig_rpm, gt_rpm]

                method_rpms.setdefault(cur_method, []).append((rpms))

    fn = 'rpms.pkl'
    # Save the results of the applied methods
    with open(results_dir + fn, 'wb') as fp:
        pickle.dump(method_rpms, fp)
        print('> Metrics saved!\n')

def print_metrics(results_dir):
    """
    Print metrics from saved RPMs.

    Parameters:
        results_dir (str): Directory containing RPMs.
    """

    fn = 'rpms.pkl'

    # Load the RPMs
    with open(results_dir + fn, 'rb') as f: 
        rpms = pickle.load(f)

    t = PrettyTable(['Method', 'RMSE', 'MAE', 'MAPE', 'PCC', 'CCC'])

    # For each extracting method...
    for method, metrics_value in rpms.items():
	
        bpmsEst = np.stack([np.squeeze(metric[0][0]) for metric in metrics_value])[np.newaxis,:]
        bpmsGT = np.stack([np.squeeze(metric[1][0]) for metric in metrics_value])	
        rmse = RMSEerror(bpmsEst, bpmsGT)
        mae = MAEerror(bpmsEst, bpmsGT)
        mape = MAPEerror(bpmsEst, bpmsGT)
        pcc = PearsonCorr(bpmsEst, bpmsGT)
        ccc = LinCorr(bpmsEst, bpmsGT)			
        vals = [rmse, mae, mape, pcc, ccc]

        t.add_row([method] + vals)

    print(t)


def main(argv):
    """
    Main function. 

    Parameters:
        argv (list): Command line arguments.
        
        0: Estimate signals (0 default), 1: Extract RPMs (Respiratory per minute), 2: Compute and print error metrics
    """

	# Define the path where to save results
    what = 0
    results_dir = './estimates/'
    video_type = 'XYZ'
    model_name = 'CliffPhys30_d'

    opts, args = getopt.getopt(argv,"ha:d:t:m:",["action=","dir=", "type=", "model="])
    for opt, arg in opts:
        if opt == '-h':
            print ('run_all.py -a <action> -d <results_dir>')
            sys.exit()
        elif opt in ("-a", "--action"):
            what = int(arg)
        elif opt in ("-d", "--dir"):
            results_dir = arg
        elif opt in ("-t", "--type"):
            video_type = arg
        elif opt in ("-m", "--model"):
            model_name = arg
    print ('Action is ', what)
    print ('Results dir is ', results_dir)	
    print ('Video type is ', video_type)	
    
    if what == 0:
        """
        Action 0:
            1) Initialize a dataset, comprising of 'RGB' or 'XYZ' (depth scalar + optial flow 2D vector field) videos. 
            To use the CliffPhys family of models, 'RGB' videos will be converted into 'XYZ' ones. Pre-generated 'XYZ' videos 
            are available in the ./data/bp4d_XYZ/ folder to bypass the need for monocular depth estimation and optical 
            flow extraction.

                DEMO:       for 'RGB' input:   datasets.BP4D()          for 'XYZ' input: datasets.BP4D_XYZ()

            2) Initialize a list of methods for extracting/predicting the respiratory wavefor from videos belonging to the given dataset.
            Method CliffPhys30 will process data configurations comprising the optical flow only, CliffPhys30_d will process 
            also the depth information.

                DEMO:       CliffPhys('CliffPhys30_d', 'PT-scamps_XYZ_FT-ALIGN_cohface_XYZ')
        """

        # Initialize a dataset
        dataset = datasets.BP4D_XYZ('raft_small-midas') 

        # Initialize a list of methods
        methods = [CliffPhys(model_name, 'PT-scamps_XYZ_FT-cohface_XYZ')] 

        # Predict respiratory waveform
        extract_respiration(dataset, methods, video_type, results_dir)
    
    elif what == 1:
        """
        Action 1:
            Extract the RPM (Respiratory Per Minute) from the provided results directory.

            DEMO:       python run_all.py -a 1 -d './estimates/' 
        """

        # Extract RPM from GT and predicted respiratory waveform
        extract_rpms(results_dir, visualize=True)
    
    elif what == 2:
        """
        Action 2:
            Print the computed metrics from the specified results directory.

            DEMO:       python run_all.py -a 2 -d './estimates/' 
        """

        # Evaluate model
        print_metrics(results_dir)

if __name__ == "__main__":
	main(sys.argv[1:])
