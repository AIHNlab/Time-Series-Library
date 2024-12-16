#import ace_tools as tools
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def parse_multiple_datasets(file_path, datasets, forecast_horizon):
    combined_data = {}
    
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    
    for dataset in datasets:
        data = {}
        # Iterate through the lines in sets of two
        for i in range(0, len(lines), 2):
            model_line = lines[i]
            metrics_line = lines[i + 1]
            
            # Exact match check using f-string
            exact_pattern = f"long_term_forecast_{dataset}_{forecast_horizon}"
            if exact_pattern in model_line:
                # Extract model name between last _ and next _
                parts = model_line.split('_')
                for j, part in enumerate(parts):
                    if part == dataset:
                        model_name = parts[j+2]  # Take model name after dataset and forecast_horizon
                        break
                
                seq_len_match = re.search(r"_sl(\d+)_", model_line)
                seq_len = int(seq_len_match.group(1)) if seq_len_match else None
                mse_match = re.search(r"mse:([\d.]+)", metrics_line)
                mse = float(mse_match.group(1)) if mse_match else None
                
                #if model_name in ["iTimesformerCyclicAttnMLPTrend"]:
                #    continue

                if model_name not in data:
                    data[model_name] = {}
                if seq_len is not None:
                    data[model_name][seq_len] = mse
        
        combined_data[dataset] = data
    
    # Rest of the function remains the same
    dfs = []
    for dataset, data in combined_data.items():
        df = pd.DataFrame(data).T.sort_index(axis=1)
        df.columns = [f"{col} timesteps" for col in df.columns]
        dfs.append(df)
    
    combined_df = pd.concat(dfs, keys=datasets, names=['Dataset', 'Models'])
    return combined_df

def plot_multiple_datasets_mse(df, forecast_horizon):
    # Create dataset separator rows
    datasets = df.index.get_level_values('Dataset').unique()
    new_data = []
    new_labels = []
    bold_cells = []
    current_row = 0
    
    for dataset in datasets:
        # Add dataset row
        new_data.append([''] * len(df.columns))
        new_labels.append(f'=== {dataset} ===')
        current_row += 1  # Account for the separator row
        
        # Add model rows for this dataset
        dataset_data = np.around(df.loc[dataset].values, decimals=5)
        dataset_labels = [f"{model}" for model in df.loc[dataset].index]
        num_models = dataset_data.shape[0]
        
        # Find best performing models per timestep, ignoring columns with all NaNs
        for col_idx in range(len(df.columns)):
            col_data = dataset_data[:, col_idx]
            if np.all(np.isnan(col_data)):
                continue  # Skip this column
            min_idx = np.nanargmin(col_data)
            bold_cells.append((current_row + min_idx, col_idx))
        
        new_data.extend(dataset_data)
        new_labels.extend(dataset_labels)
        current_row += num_models  # Account for the model rows
    
    # Adjust figure size for additional rows
    fig, ax = plt.subplots(figsize=(12, len(new_data)*0.4))
    ax.set_axis_off()
    
    table = ax.table(cellText=new_data,
                     rowLabels=new_labels,
                     colLabels=df.columns,
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Highlight dataset rows
    for idx, label in enumerate(new_labels):
        if label.startswith('==='):
            for col in range(-1, len(df.columns)):
                cell = table[idx+1, col]
                cell.set_facecolor('#e6e6e6')
                cell.set_text_props(weight='bold')
    
    # Mark best performing models in bold
    for row_idx, col_idx in bold_cells:
        cell = table[row_idx+1, col_idx]
        cell.set_text_props(weight='bold')
    
    plt.title(f'MSE Results for forecast length: {forecast_horizon}')
    plt.tight_layout()
    plt.show()
    
def total_best_models_per_timestep(df_combined):
    # Initialize a list to store the results
    results = []
    # Iterate over each dataset
    for dataset in df_combined.index.levels[0]:
        # Select data for the current dataset
        df_dataset = df_combined.loc[dataset]
        # Find the model with the lowest MSE for each timestep
        best_models = df_dataset.idxmin()
        # Record the best model for each timestep
        for timestep, model in best_models.items():
            results.append({'Dataset': dataset, 'Model': model, 'Timestep': timestep})
    # Create a DataFrame from the results
    df_results = pd.DataFrame(results)
    # Count the number of times each model is the best at each timestep across all datasets
    counts = df_results.groupby(['Timestep', 'Model']).size().unstack(fill_value=0)
    # Reset index to make 'Timestep' a column
    counts = counts.reset_index()
    # Extract numerical values from 'Timestep' and convert to integers
    counts['TimestepNum'] = counts['Timestep'].str.extract('(\d+)').astype(int)
    # Sort by 'TimestepNum'
    counts = counts.sort_values('TimestepNum')
    # Set 'Timestep' back as index and drop 'TimestepNum'
    counts = counts.set_index('Timestep').drop('TimestepNum', axis=1)
    return counts

# Example usage:
#file_path = "./result_long_term_forecast_UTSD_3run.txt"
#datasets = ["KDDCup2018", "ERA5Surface", "AustrailianElectricityDemand", "BenzeneConcentration", "MotorImagery", "TDBrain", "LondonSmartMeters", "AustraliaRainfall", "BeijingAir", "PedestrianCounts"]  # Add your datasets here

file_path = "./result_long_term_forecast_Benchmark_2run.txt"
datasets = ["Electricity", "Weather", "Traffic", "ETTh2", "ETTm2"] 

forecast_horizon = 96

# Parse and plot
df_combined = parse_multiple_datasets(file_path, datasets, forecast_horizon)
print(df_combined)
print(total_best_models_per_timestep(df_combined))
plot_multiple_datasets_mse(df_combined, forecast_horizon)

#tools.display_dataframe_to_user(name="Parsed Forecast Results by Cycles (Linebreak Handling)", dataframe=df_cycles_linebreaks)