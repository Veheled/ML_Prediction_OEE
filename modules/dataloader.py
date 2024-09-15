from azureml.core import Workspace, Dataset, Datastore

import pandas as pd
import numpy as np
import os
from typing import List
from datetime import datetime

class DataLoader():

    def __init__(self, prod: bool, datafolder_path: str, reload_list: List[str], raw_data_folder_path: str):
        self.prod = prod # Productive Datastore or Development Datastore Boolean
        self.datafolder_path = datafolder_path # Path to folder from which to load datafiles in datastorage
        self.reload_list = reload_list # List of filenames that dataloader should reload
        self.raw_data_folder_path = raw_data_folder_path # Path to folder in which to load raw data

    # Function to get datastore object from azure subcription (datastore setup in Azure ML environment)
    def __get_subscription_datastore(self):
        """
            Return subcription datastore
        """
        subscription_id = 'your_subscription_id'
        resource_group = 'your_resource_group'
        workspace_name = 'your_workspace_name'
        
        # Initialize Azure ML Workspace instance
        workspace = Workspace(subscription_id, resource_group, workspace_name)

        # Depending on dataloader setting choose different datastorage
        if self.prod:
            datastore = Datastore.get(workspace, "your_prod_datastore")
        else:
            datastore = Datastore.get(workspace, "your_dev_datastore")
        
        return datastore

    def __get_parquet_table_from_datastore(self, datastore: Datastore, filepath: str):
        dataset = Dataset.Tabular.from_parquet_files(path=(datastore, filepath))
        dataframe = dataset.to_pandas_dataframe() 
        return dataframe

    def run(self):
        print('Attempt to initialize datastore', flush=True)
        datastore = self.__get_subscription_datastore()
        print('Datastore connected!', flush=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    
        for file in self.reload_list:
            print(f'Attempt to load { self.datafolder_path+file}', flush=True)
            raw_df = self.__get_parquet_table_from_datastore(datastore, self.datafolder_path+file)
            raw_df.to_parquet(f'{self.raw_data_folder_path}{timestamp}_{file}')
            print(f'successfully to loaded {len(raw_df)} from {self.datafolder_path+file}', flush=True)

