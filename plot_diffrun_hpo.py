import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from result_analyser_hpo import parse_results


def generate_scatter_plot(df1, df2, figsize=(8,8), is_mse=True):

    # Find common datasets, models, and prediction lengths
    common_datasets = df1.index.intersection(df2.index)
    common_columns = df1.columns.intersection(df2.columns)
    
    # Subset data to only include common elements
    filtered_df1 = df1.loc[common_datasets, common_columns]
    filtered_df2 = df2.loc[common_datasets, common_columns]
    
    # Flatten data for scatter plot
    x = filtered_df1.values.flatten()
    y = filtered_df2.values.flatten()
    
    # Create scatter plot
    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.7)
    
    # Plot y = x line
    xpoints = ypoints = plt.xlim()
    plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1.0, scalex=False, scaley=False)

    # Add labels, title, and legend
    var = "MSE" if is_mse else "MAE"
        
    plt.xlabel(f"{var}")
    plt.ylabel(f"{var}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
datasets = ["BenzeneConcentration", "MotorImagery", "TDBrain", "BeijingAir", 
             "ETTh1", "ETTm1", "ETTh2", "ETTm2", "Weather", "Exchange"]
forecast_horizons = [96, 192, 336, 720]
is_mse = True

file_path1 = "./results_2ndhpo.txt"
file_path2 = "./results_hpo_lorenzo.txt"

mean_results1, std_results1 = parse_results(file_path1, datasets, forecast_horizons, is_mse=is_mse)
mean_results2, std_results2 = parse_results(file_path2, datasets, forecast_horizons, is_mse=is_mse)


generate_scatter_plot(mean_results1, mean_results2, is_mse=is_mse)
