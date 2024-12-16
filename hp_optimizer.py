import argparse
import os
import torch
import optuna
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from utils.print_args import print_args
import random
import numpy as np
import pandas as pd
from datetime import datetime

# Global to track last file
last_trial_file = None

def save_trials_callback(study, trial):
    """Save trial results to CSV after each trial"""
    global last_trial_file
    
    # Create results directory
    os.makedirs('hp_results', exist_ok=True)
    
    # Delete previous file
    if last_trial_file and os.path.exists(last_trial_file):
        os.remove(last_trial_file)
    
    # Save new file
    df = study.trials_dataframe()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    last_trial_file = f'hp_results/trials_{args.model}_{args.model_id}.csv'
    df.to_csv(last_trial_file, index=False)

def select_experiment(args):
    # Select the experiment class based on task_name
    if args.task_name == 'long_term_forecast':
        return Exp_Long_Term_Forecast(args)
    elif args.task_name == 'short_term_forecast':
        return Exp_Short_Term_Forecast(args)
    elif args.task_name == 'imputation':
        return Exp_Imputation(args)
    elif args.task_name == 'anomaly_detection':
        return Exp_Anomaly_Detection(args)
    elif args.task_name == 'classification':
        return Exp_Classification(args)
    else:
        return Exp_Long_Term_Forecast(args)

# Define the objective function
def objective(trial):
    try:
        # New/modified hyperparameters from paper
        if any(dataset in args.model_id for dataset in ["ETTh2", "Electricity", "Traffic"]):
            args.seq_len = trial.suggest_categorical('seq_len', [24, 72, 168, 336, 480])
        if any(dataset in args.model_id for dataset in ["Weather"]):
            args.seq_len = trial.suggest_categorical('seq_len', [144, 288, 576])
        if any(dataset in args.model_id for dataset in ["ETTm2"]):
            args.seq_len = trial.suggest_categorical('seq_len', [96, 192, 384, 480])
        #args.seq_len = trial.suggest_categorical('seq_len', [24, 96, 192, 336, 512])
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-2, log=True)
        args.e_layers = trial.suggest_int('e_layers', 1, 5)
        args.d_model = trial.suggest_int('d_model', 16, 512, step=16)
        args.train_epochs = trial.suggest_int('train_epochs', 10, 100)

        # Suggest hyperparameters
        #args.d_model = trial.suggest_int('d_temp', 128, 1024, step=128)
        #args.n_heads = trial.suggest_int('n_heads', 1, 8)
        #args.e_layers = trial.suggest_int('e_layers', 1, 3)
        #args.d_layers = trial.suggest_int('d_layers', 1, 4)
        args.d_ff = trial.suggest_int('d_ff', 256, 2048, step=256)
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        #args.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        #args.dropout = trial.suggest_float('dropout', 0.0, 0.5)
        #args.factor = trial.suggest_int('factor', 1, 5)

        # iTimesformer parameters
        #args.main_cycle = trial.suggest_int('main_cycle', 1, 24)  
        args.d_temp = trial.suggest_int('d_temp', 128, 1024, step=128)
        #args.x_mark_size = trial.suggest_int('x_mark_size', 0, 8)
        args.full_mlp = trial.suggest_categorical('full_mlp', [True, False])
        args.model_trend = trial.suggest_categorical('model_trend', [True, False])    

        exp = select_experiment(args)
        setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'

        print(f"Starting trial {trial.number}")
        exp.train(setting)

        # Validate
        vali_data, vali_loader = exp._get_data(flag='val')
        criterion = exp._select_criterion()
        val_loss = exp.vali(vali_data, vali_loader, criterion)
        print(f"Trial {trial.number} validation loss: {val_loss}")

        return val_loss

    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned")
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {str(e)}")
        # Log error details
        with open('hp_results/failed_trials.log', 'a') as f:
            f.write(f"Trial {trial.number} failed:\n")
            f.write(f"Parameters: {trial.params}\n")
            f.write(f"Error: {str(e)}\n\n")
        
        # Clean up any leftover files/resources
        setting = f'hp_search_{args.model_id}_{args.model}/trial_{trial.number}'
        cleanup_path = os.path.join('./checkpoints/', setting)
        if os.path.exists(cleanup_path):
            import shutil
            shutil.rmtree(cleanup_path)
            
        # Return worst possible value to ensure failed trials aren't selected
        return float('inf')

if __name__ == '__main__':
    # Set random seeds for reproducibility
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Hyperparameter Optimization')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=0, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=48,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False, 
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # iTimesformer
    parser.add_argument('--main_cycle', type=int, default=24, help='main cycle')
    parser.add_argument('--d_temp', type=int, default=1024, help='bottleneck for dimensionality of time attention')
    parser.add_argument('--full_mlp', action='store_true', help='Use MLP layers in iTransformer style')
    parser.add_argument('--model_trend', action='store_true', help='Model trend with a linear layer')
    parser.add_argument('--x_mark_size', type=int, default=0, help='size of external features')
    parser.add_argument('--results_file', type=str, help='alternative file to write final results to')    

    # UTSD dataset
    parser.add_argument('--stride', type=int, default=1, help='stride of the sliding window (just for UTSD dataset)')
    parser.add_argument('--split', type=float, default=0.9, help='training set ratio')

    args = parser.parse_args()

    # Parse arguments
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() else False

    # Handle multi-GPU setup if needed
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # Ensure 'is_training' is set to True
    args.is_training = 1

    # Initialize Optuna study
    study = optuna.create_study(direction='minimize')

    # Start the optimization
    study.optimize(objective, n_trials=40, callbacks=[save_trials_callback])

    # Output the best hyperparameters
    print('Number of finished trials:', len(study.trials))
    print('Best trial:')
    trial = study.best_trial

    print('  Value:', trial.value)
    print('  Params:')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

    # Retrain the model with the best hyperparameters
    for param_name, param_value in trial.params.items():
        setattr(args, param_name, param_value)

    exp = select_experiment(args)
    setting = f'hp_search_{args.model_id}_{args.model}'
    exp.train(setting)

    # Test the model
    exp.test(setting)