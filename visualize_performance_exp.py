import pandas as pd
import numpy as np

import tkinter as tk
from tkinter import ttk

def import_results(path: str) -> pd.DataFrame:
    
    with open(path, "r") as file:
        lines = file.readlines()

    experiments, has_errors, error_types, times, gpus, cpus = [], [], [], [], [], []

    for i in range(0, len(lines), 3):
        experiment, metrics = lines[i].strip(), lines[i+1].strip()
        
        has_error = metrics.split(", ")[0].split(":")[1] == "True"
        error_type = metrics.split(", ")[1].split(":")[1]
        if has_error:
            time, gpu, cpu = np.nan, np.nan, np.nan
        else:
            time = float(metrics.split(", ")[2].split(":")[1])
            gpu = float(metrics.split(", ")[3].split(":")[1])
            cpu = float(metrics.split(", ")[4].split(":")[1])

        experiments.append(experiment)
        has_errors.append(has_error)
        error_types.append(error_type)
        times.append(time)
        gpus.append(gpu)
        cpus.append(cpu)
        
    # extract model and dataset from 'experiments'
    datasets = [experiment.split("_")[3] for experiment in experiments]
    models = [experiment.split("_")[5] for experiment in experiments]

    results = pd.DataFrame({
        "dataset": datasets,
        "model": models,
        "time": times,
        "gpu": gpus,
        "cpu": cpus,
        "error_type": error_types,
    })
    
    # set multiindex 'dataset' and 'model'
    results.set_index(["model", "dataset"], inplace=True)
    results.sort_index(inplace=True)    
    
    return results

def show_dataframe_with_index(df, multiindex=None):
    # Sort multiindex DataFrame by sort_by
    if multiindex:
        df = df.reset_index().set_index(multiindex)
        df = df.sort_index()
    
    # Create a new Tkinter window
    window = tk.Tk()
    window.title("DataFrame Viewer with Multi-Level Index")

    # Flatten the multi-level index for display
    flattened_index = df.index.to_frame(index=False)
    df_with_index = pd.concat([flattened_index, df.reset_index(drop=True)], axis=1)

    # Create a Treeview widget
    tree = ttk.Treeview(window, columns=list(df_with_index.columns), show="headings")

    # Add column headings
    for col in df_with_index.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Add rows to the Treeview
    for row in df_with_index.itertuples(index=False):
        tree.insert("", tk.END, values=row)

    # Pack the Treeview widget
    tree.pack(expand=True, fill="both")

    # Start the Tkinter event loop
    window.mainloop()



results_file = "results_performance_exp.txt"

results = import_results(results_file)
results["time"] = np.ceil(results["time"]) # ceil to next integer for better visibility

# exclude datasets and models
excl_datasets = ["Electricity", "Traffic", "ERA5Surface", "ERA5Pressure"]
excl_models = ["TimeMixer"]
results = results[~results.index.get_level_values("dataset").isin(excl_datasets)]
results = results[~results.index.get_level_values("model").isin(excl_models)]


#****************************** Expressive Table **********************************

# show_dataframe_with_index(results, multiindex=["model", "dataset"])
# show_dataframe_with_index(results, multiindex=["dataset","model"])


#****************************** Pivot Table **********************************

# make table with index 'model' and columns that are the datasets (currently in multiindex). Use time as the values in the table
results_pivot= results.pivot_table(index="model", columns="dataset", values="time").astype(int)

# add index with sum
results_pivot.loc["sum"] = results_pivot.sum().astype(int)
results_pivot.sort_values(by="sum", axis=1, inplace=True)

# add column with sum time in the end
results_pivot["sum"] = results_pivot.sum(axis=1).astype(int)
results_pivot.sort_values(by="sum", inplace=True)
results_pivot = pd.concat([results_pivot.loc[results_pivot.index != "sum"], results_pivot.loc[["sum"]]])
results_pivot.loc['sum', 'sum'] = ''

show_dataframe_with_index(results_pivot)
