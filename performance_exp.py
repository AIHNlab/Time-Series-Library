import argparse
import os
import torch
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np

import time
import traceback
import psutil
from run import parse_args

TIME_EXP_CONFIG = {
    'seq_len': 720,
    'pred_len': 720,
    'e_layers': 4,
    'd_model': 512,
    'd_ff': 512,
    'batch_size': 32,
    'train_epochs': 1,
    'results_file': 'results_performance_exp.txt'
}

TIME_EXP_CONFIG_TIMEMIXER = {
    'seq_len': 720,
    'pred_len': 720,
    'e_layers': 3,
    'd_model': 128,
    'd_ff': 128,
    'batch_size': 8,
    'train_epochs': 1,
    'results_file': 'results_performance_exp.txt'
}


if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    args = parse_args()
    
    # overwrite args with worst case config
    if args.model_id == 'TimeMixer':
        for key, value in TIME_EXP_CONFIG_TIMEMIXER.items():
            setattr(args, key, value)
    else:
        for key, value in TIME_EXP_CONFIG.items():
            setattr(args, key, value)
    
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

            print('>>>>>>>measure training time on : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            try:
                # Get the process ID of the current Python program
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss  # Resident Set Size in bytes

                start_time = time.time()
                exp.train(setting)
                end_time = time.time() 
                execution_time = end_time - start_time 
                
                after_memory = process.memory_info().rss  # Peak memory usage on CPU in bytes
                peak_cpu = (after_memory - initial_memory) / (1024 ** 2) # Peak memory usage on CPU in MB
                
                peak_gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)   # Peak memory usage on GPU in MB

                if not os.path.exists(args.results_file): # make the file if it does not exist
                    with open(args.results_file, "w") as file:
                        file.write("")
                with open(args.results_file, "a") as file:  # open the file in append mode
                    file.write(f"{setting}\nerror:False, type:na, time_in_s:{execution_time:.3f}, gpu_in_mb:{peak_gpu:.3f}, cpu_in_mb:{peak_cpu:.3f}\n\n")

                print(f"{setting}\n{execution_time:.3f} seconds\n\n")
                
            except Exception as e:
                print(e)
                print('Errors occurred during training...')
                error_message = f"Error while running foo(): {str(e)}\n"
                error_details = traceback.format_exc()  # Get full traceback as a string
                if not os.path.exists(args.results_file): # make the file if it does not exist
                    with open(args.results_file, "w") as file:
                        file.write("")
                with open(args.results_file, "a") as file:
                    error_type = type(e).__name__
                    file.write(f'{setting}\nerror:True, type:{error_type}, time_in_s:na, gpu_in_mb:na, cpu_in_mb:na\n\n')

            torch.cuda.empty_cache()
    else:
        raise NotImplementedError('This script is used for measuring the memory and time it takes to train 1 epoch.')
