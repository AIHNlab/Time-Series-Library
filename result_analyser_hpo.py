import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

def parse_results(file_path, datasets, forecast_horizons, is_mse=True):
    results_mean = {}
    results_std = {}
    
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    for horizon in forecast_horizons:
        horizon_data_mean = {}
        horizon_data_std = {}
        for dataset in datasets:
            scores_dict = {}
            
            for i in range(0, len(lines), 2):
                model_line = lines[i]
                metrics_line = lines[i + 1]
                exact_pattern = f"hp-search_{dataset}_{horizon}"
                
                if exact_pattern in model_line:
                    parts = model_line.split('_')
                    model_name = next(parts[j+2] for j, part in enumerate(parts) if part == dataset)
                    
                    pattern = r"mse:([\d.]+)" if is_mse else r"mae:([\d.]+)"
                    match = re.search(pattern, metrics_line)
                    score = float(match.group(1)) if match else None
                    
                    if score is not None:
                        if model_name not in scores_dict:
                            scores_dict[model_name] = []
                        scores_dict[model_name].append(score)
            
            if scores_dict:
                means = {k: np.mean(v) for k, v in scores_dict.items()}
                stds = {k: np.std(v) if len(v) > 1 else 0 for k, v in scores_dict.items()}
                horizon_data_mean[dataset] = means
                horizon_data_std[dataset] = stds
        
        results_mean[f'H{horizon}'] = pd.DataFrame.from_dict(horizon_data_mean, orient='index')
        results_std[f'H{horizon}'] = pd.DataFrame.from_dict(horizon_data_std, orient='index')
    
    return (pd.concat({k: v for k, v in results_mean.items()}, axis=1),
            pd.concat({k: v for k, v in results_std.items()}, axis=1))

def plot_results(mean_df, std_df, forecast_horizons, is_mse=True):
    metric = 'MSE' if is_mse else 'MAE'
    fig, axes = plt.subplots(nrows=len(forecast_horizons), figsize=(15, 4*len(forecast_horizons)))
    if len(forecast_horizons) == 1:
        axes = [axes]
    
    for i, horizon in enumerate(forecast_horizons):
        ax = axes[i]
        mean_sub = mean_df.xs(f'H{horizon}', level=0, axis=1)
        std_sub = std_df.xs(f'H{horizon}', level=0, axis=1)
        
        numeric_mean = mean_sub.apply(pd.to_numeric, errors='coerce')
        min_values = numeric_mean.min(axis=1)
        
        # Format cells as mean ± std
        cell_text = numeric_mean.copy()
        for col in cell_text.columns:
            cell_text[col] = cell_text[col].map(lambda x: f'{x:.4f}') + ' ± ' + \
                            std_sub[col].map(lambda x: f'{x:.4f}')
        
        table = ax.table(
            cellText=cell_text.values,
            rowLabels=mean_sub.index,
            colLabels=mean_sub.columns,
            cellLoc='center',
            loc='center'
        )
        
        # Bold the best results
        for row_idx, row in enumerate(numeric_mean.values):
            for col_idx, val in enumerate(row):
                if pd.notnull(val) and val == min_values[row_idx]:
                    cell = table[row_idx + 1, col_idx]
                    cell.set_text_props(weight='bold')
        
        ax.set_axis_off()
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5, 1.8)
        
        # Set the title higher above the table
        ax.set_title(f'{metric} Results for Horizon: {horizon}', pad=30)
    
    # Adjust spacing between plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Add extra space for titles if needed
    plt.show()

# Example usage:
file_path = "./results_hpo_lorenzo.txt"
datasets = ["BenzeneConcentration", "MotorImagery", "TDBrain", "BeijingAir", 
            "ETTh1", "ETTm1", "ETTh2", "ETTm2", "Weather", "Exchange"]
forecast_horizons = [96, 192, 336, 720]

mean_results, std_results = parse_results(file_path, datasets, forecast_horizons)
plot_results(mean_results, std_results, forecast_horizons)