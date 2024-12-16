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
from run import parse_args

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

    args = parse_args()

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