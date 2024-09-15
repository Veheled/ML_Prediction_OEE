import pandas as pd
import numpy as np
import os
from datetime import datetime
import re
import joblib

# Function to 
def load_latest_dataset_from_storage(directory: str, keyword: str):
    latest_file = None
    latest_time = datetime.min
    date_pattern = re.compile(r'\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}')
    for file in os.listdir(directory):
        if keyword in file:
            match = date_pattern.search(file)
            if match:
                try:
                    file_time = datetime.strptime(match.group(), '%Y-%m-%d_%H_%M_%S')
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file
                except ValueError as e:
                    print("Error parsing time from filename:", file, "; Error:", e, flush=True)
    return pd.read_parquet(os.path.join(directory, latest_file)) if latest_file else None

def load_best_model_from_storage(model_directory: str, target: str):
    best_rmse = float('inf')
    best_model = None
    for file in os.listdir(model_directory):
        # Adjusted to check if the target is in the filename and the file is a joblib file
        if target in file and file.endswith('.joblib'):
            # Further splitting by 'model_val_rmse' to isolate the rmse value correctly
            try:
                rmse_str = file.split('_model_val_rmse_')[1]
                rmse = float(rmse_str.split('_')[0])
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = file
            except Exception as e:
                print(f"Error processing file {file}: {e}", flush=True)
    return joblib.load(os.path.join(model_directory, best_model)) if best_model else None

def delete_all_files_in_folder(folder_path: str):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.", flush=True)
        return
    
    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Check if it is a file (not a directory)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}", flush=True)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}", flush=True)
