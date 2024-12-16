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
from time import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from run import parse_args


def run_experiment(args):
    
    print(f'Sequence length {args.seq_len} - Variables {args.c_out} - Model: {args.model}')
    args.features = 'M'
        
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
    
    exp = Exp(args)
    iters = args.train_epochs
    
    model = exp._build_model().cuda()

    if args.is_training:
        optimizer = exp._select_optimizer()
        criterion = exp._select_criterion()
    else:
        model.eval()

    input_data = torch.randn(args.batch_size, args.seq_len, args.c_out).cuda()
    temp_data = torch.randn(args.batch_size, args.seq_len, 3).cuda()
    target = torch.randn(args.batch_size, args.pred_len, args.c_out).cuda()

    num_proc_seq = args.batch_size * iters

    start = time()
    for it in tqdm(range(iters)):
        
        torch.cuda.reset_peak_memory_stats()
        
        if args.is_training:
            output = model(input_data, None, None, None)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(input_data, None, None, None)
            
        peak_memory = torch.cuda.max_memory_allocated()
        peak_memory *= 1e-6  # map to MB
        
    end = time()

    seq_per_second = num_proc_seq / (end - start)
    
    return {"seq_per_second": seq_per_second, "peak_memory": peak_memory}  
    
    
if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args = parse_args()
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.use_gpu = True if torch.cuda.is_available() else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

        
    # Assuming args is a predefined argument object
    models = args.models
    seq_lens = args.seq_lens
    c_outs = args.c_outs
    
    if (seq_lens is None) and (c_outs is None):
        raise('Specify either seq_lens or c_outs')
    elif not(seq_lens is None) and not(c_outs is None):
        raise('Specify only one between seq_lens and c_outs')
    elif not(seq_lens is None):
        x = seq_lens
    elif not(c_outs is None):
        x = c_outs
    
        
    KEYS = ['seq_per_second', 'peak_memory']
    KEYS_PLOT = ['Training Speed (seq/s)', 'Peak Memory Usage (MiB)']
    if args.is_training == 0:
        KEYS_PLOT[0] = 'Inference Speed (seq/s)'
    
    # Initialize a dictionary to store results for each metric across models and sequence lengths
    results = {key: np.zeros((len(x), len(models))) for key in KEYS}
    
    # Loop through sequence lengths and models
    for i, v in enumerate(x):
        for j, m in enumerate(models):
            if x == seq_lens:
                args.seq_len = v
            if x == c_outs:
                args.c_out = v
            args.model = m
    
            # Run the experiment and get the result dictionary
            experiment_out = run_experiment(args)
    
            # Store the results in the respective arrays
            for key in experiment_out:
                results[key][i, j] = experiment_out[key]
    
    # Plotting results
    for ik, (key, data) in enumerate(results.items()):
        plt.figure(figsize=(8, 6))
        for j, m in enumerate(models):
            plt.plot(x, data[:, j], label=f"{m}", marker='o')
        
        if x == seq_lens:
            plt.title(f"{args.c_out} Variables")
            plt.xlabel("Sequence Length (L)")
        if x == c_outs:
            plt.title(f"{args.seq_len} Sequence Length")
            plt.xlabel("Variables (C)")
            
        plt.xticks(x, labels=[str(v) for v in x])
        plt.ylabel(KEYS_PLOT[ik])
        plt.legend()
        plt.grid(True)
        if x == seq_lens:
            plt.savefig(f"{key}_vs_seq_len.png")
        if x == c_outs:
            plt.savefig(f"{key}_vs_c_out.png")
        plt.show()
    
#     # TODO
    
#     from calflops import calculate_flops
#     from torchvision import models

#     model = models.alexnet()
#     batch_size = 1
#     input_shape = (batch_size, 3, 224, 224)
#     flops, macs, params = calculate_flops(model=model, 
#                                           input_shape=input_shape,
#                                           include_backPropagation=True,
#                                           output_as_string=False,
#                                           output_precision=4)
#     print("Alexnet FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
#     ######
    



