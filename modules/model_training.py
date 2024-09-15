import glob
import os
import warnings
from typing import List

# Import general ds packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from joblib import dump

# Sklearn Modules
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim

# Load necessary models
from modules.utils import load_latest_dataset_from_storage
from modules.training_pipeline_steps import feature_column_selection, feature_encoding, train_test_split, feature_numeric_normalization,\
                                            initialize_model, optimize_model_parameters, initialize_model, train_model_kfold_cross_val, validate_model, RegressionNN,\
                                            train_neural_network, store_models, optimize_neural_network_parameters

class Model_Trainer():

    def __init__(self, 
                 integrated_data_path: str, # Path to the storage of integrated data files
                 feature_data_path: str, # Path to the storage of feature data files
                 raw_data_folder_path: str, # Path to the storage of raw data files
                 preprocessed_data_folder_path: str, # Path to the storage of preprocessed data files
                 frontend_reference_folder_path: str, # Path to the storage of frontend reference data files
                 feature_list: List[str], ### Array of features That should be used in the training loop
                 model_targets: List[str], ### Array of models That should be trained for training loop
                 models_to_train: List[str], ### Array of models That should be trained for training loop
                 validation_ratio: float, ### Float between 0 and 1 that determines the dataset split in training and validation
                 scaling_enabled: bool, ### Boolean to determine if scaling is enabled for all models or not
                 product_encoding_method: str, ### method in string to use for product encoding - ordinal or nominal
                 save_models: bool, ### Whether or not to save the models trained in the model storage
                 model_optimization_do: bool, ### Whether or not the main loop performs grid/random search optimization
                 optimization_mode: str, # gridsearch or randomsearch,
                 model_test_name: str # name to be put in front of models
                 ) -> None:
        """
        Model Trainer
        """
        # General Trainer Settings
        self.integrated_data_path = integrated_data_path
        self.feature_data_path = feature_data_path
        self.raw_data_folder_path = raw_data_folder_path
        self.preprocessed_data_folder_path = preprocessed_data_folder_path
        self.frontend_reference_folder_path = frontend_reference_folder_path
        self.feature_list = feature_list      
        self.model_targets = model_targets 
        self.validation_ratio = validation_ratio
        self.scaling_enabled = scaling_enabled
        self.product_encoding_method = product_encoding_method
        self.models_to_train = models_to_train
        self.save_models = save_models
        self.model_optimization_do = model_optimization_do
        self.optimization_mode = optimization_mode
        self.model_test_name = model_test_name

        ### Definition of used error functions and their names for charts / printouts
        self.error_function = mean_squared_error
        self.ef_name = "RMSE"
        self.error_function_2 = mean_absolute_error
        self.ef_name_2 = "MAE"

        ## Optimization trials for random search or otuna
        self.optimization_iters = 25

        ## Neural Network Settings:
        self.num_epochs = 2000
        self.learning_rate = 0.001
        self.early_stopping_patience = 100

        # Result list for all trained models
        self.result_model_performance = [] 
        
        # Load Training Data
        self.__load_training_data()
        print('Training Data Loaded!')

        # Store preprocessed reference datasets
        self.__store_preprocessed_data()
        print('Data preprocessed and stored as reference files!')
        
    def __load_training_data(self) -> None:
        ### Read latest Fastec datafile by timestamp from folder
        keyword = "Fastec_Formate_Dataset"   # Pattern to match the files
        latest_integrated_data_file = load_latest_dataset_from_storage(self.integrated_data_path, keyword)
        keyword = "ProductChange_Category_Features"   # Pattern to match the files
        latest_order_change_feature_file = load_latest_dataset_from_storage(self.feature_data_path, keyword)
        keyword = "Product_HistoricTimes_Features"   # Pattern to match the files
        latest_historic_times_feature_file = load_latest_dataset_from_storage(self.feature_data_path, keyword)

        latest_integrated_data_file['ProductChange'] = latest_integrated_data_file['Previous_ProductCode'] + "-" + latest_integrated_data_file['ProductCode']
        latest_integrated_data_file = latest_integrated_data_file.merge(latest_order_change_feature_file, on='ProductChange', how='left')
        latest_integrated_data_file = latest_integrated_data_file.merge(latest_historic_times_feature_file, on='ProductCode', how='left')

        self.training_dataset = latest_integrated_data_file

    def __store_preprocessed_data(self) -> None:
        # Current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

        product_reference = self.training_dataset
        product_reference = product_reference[["ProductCode", "FS_Breite", "FS_Länge", "FS_Tiefe", "PBL_Länge", "PBL_Breite", "Tuben_Durchmesser", "Tuben_Länge"]]\
            .drop_duplicates().fillna(0)
        df_DimMdaOperation =  load_latest_dataset_from_storage(self.raw_data_folder_path, 'DimMdaOperation')
        product_reference = product_reference.merge(df_DimMdaOperation[['ProductCode', 'ProductDescription', 'ProductGroupCode']].drop_duplicates().dropna(),
                                                        'left',
                                                        on = "ProductCode"
                                                    ).fillna('Unbekannt')
        product_reference = product_reference.merge(self.training_dataset[['ProductCode', 'CALC_WIRKSTOFF', 'CALC_ALUFOLIE', 'CALC_PACKGROESSE']].drop_duplicates(),
                                                    'left',
                                                     on='ProductCode')
       
        filtered_training_data, targets = feature_column_selection(self.training_dataset, self.feature_list, ['OEE', 'PERF', 'AVAIL', 'QUAL', 'DT', 'APT', 'PBT'])
              
        product_encoding = feature_encoding(filtered_training_data[['ProductCode']].drop_duplicates(), False, ['ProductCode'])

        line_encoding = feature_encoding(filtered_training_data[['Code']].drop_duplicates(), False, ['Code'])

        encoded_training_data = feature_encoding(filtered_training_data, True, ['ProductCode', 'Code', 'CALC_WIRKSTOFF', 'CALC_ALUFOLIE'])
        encoded_normalized_training_data = feature_numeric_normalization(encoded_training_data, False)

        encoded_training_data.to_parquet(f'{self.preprocessed_data_folder_path}{timestamp}_Encoded_Feature_Dataset.parquet')
        encoded_normalized_training_data.to_parquet(f'{self.preprocessed_data_folder_path}{timestamp}_Encoded_Normalized_Feature_Dataset.parquet')

        product_reference.to_parquet(f'{self.frontend_reference_folder_path}{timestamp}_Product_Reference.parquet')
        product_encoding.to_parquet(f'{self.frontend_reference_folder_path}{timestamp}_ProductEncoding_Reference.parquet')
        line_encoding.to_parquet(f'{self.frontend_reference_folder_path}{timestamp}_LineEncoding_Reference.parquet')

    def run_training_pipeline(self, verbose_train, verbose_test) -> None:
        warnings.filterwarnings('ignore', category=FutureWarning)

        for target_value_column in self.model_targets:
            for model in self.models_to_train:
                ### HANDLE FEATURE PREPROCESSING AND SELECTION
                features, target_value = feature_column_selection(self.training_dataset, self.feature_list, [target_value_column])  ## Select and clean feature and target columns from original dataset
                string_columns = features.select_dtypes(include=['object']).columns.tolist()
                features = feature_encoding(features, True, string_columns, self.product_encoding_method) ## Label Encode remaining categorical features

                ### SETUP TRAIN TEST VALIDATION SPLIT
                X = features
                y = target_value

                # Split into training and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.validation_ratio, random_state=42)

                # Init Grid Search parameters
                best_parameters  = None

                ## Switch Neural Network vs. Standard Methods
                if model != 'NN':
                    # APPLY NORMALIZATION TO TRAINING DATA ONLY
                    if model == 'svr':
                        if self.model_test_name == 'SCALING_TEST':
                            X_train = feature_numeric_normalization(X_train, self.scaling_enabled) ## If True: Apply feature scaling on numeric features (Zero Centered, Normalized) -> Zero-Center-Normalization
                            X_test = feature_numeric_normalization(X_test, self.scaling_enabled) ## If True: Apply feature scaling on numeric features (Zero Centered, Normalized) -> Zero-Center-Normalization
                        else:
                            X_train = feature_numeric_normalization(X_train, True) ## If True: Apply feature scaling on numeric features (Zero Centered, Normalized) -> Zero-Center-Normalization
                            X_test = feature_numeric_normalization(X_test, True) ## If True: Apply feature scaling on numeric features (Zero Centered, Normalized) -> Zero-Center-Normalization
                    else:
                        X_train = feature_numeric_normalization(X_train, self.scaling_enabled) ## If True: Apply feature scaling on numeric features (Zero Centered, Normalized) -> Zero-Center-Normalization

                    ### DO GRIDSEARCH FOR MODEL OPTIMIZATION IF ENABLED
                    if self.model_optimization_do:
                        ### init Gridsearch model
                        gs_model = initialize_model(model, None)

                        # Further sample the training data for grid search
                        # Adjust the test_size to control the fraction used for grid search (e.g., 0.5 for 50%, 0.33 for 33% of total dataset for gridsearch)
                        #X_train_rem, X_gridsearch, y_train_rem, y_gridsearch = train_test_split(X_train, y_train, test_size=1.0, random_state=42)

                        best_parameters, best_score = optimize_model_parameters(self.optimization_mode, gs_model, model, X_train, y_train.values.ravel(), self.optimization_iters)
                    
                    ### INITALIZE FINAL MODELS
                    init_model = initialize_model(model, best_parameters)

                    ### TRAIN FINAL MODELS
                    train_error, train_error2, scores, all_actuals, all_predictions = train_model_kfold_cross_val(init_model, model, target_value_column, 5, X_train, y_train, self.ef_name, self.error_function, self.ef_name_2, self.error_function_2, verbose_train)

                    ### VALIDATE FINAL MODELS
                    X_val = X_test.copy()
                    y_val = y_test.copy()
                    X_val.fillna(0, inplace=True)

                    val_error, val_error2, y_pred = validate_model(init_model, model, target_value_column, X_val, y_val, self.ef_name, self.error_function, self.ef_name_2, self.error_function_2, verbose_test)
                else:
                    # APPLY NORMALIZATION TO TRAINING DATA ONLY
                    if self.model_test_name == 'SCALING_TEST':
                        X_train = feature_numeric_normalization(X_train, self.scaling_enabled)
                        X_test = feature_numeric_normalization(X_test, self.scaling_enabled)
                    else:
                        X_train = feature_numeric_normalization(X_train, True)
                        X_test = feature_numeric_normalization(X_test, True)

                    X_val = X_test.copy()
                    y_val = y_test.copy()

                    # Hyperparameter optimization
                    if self.model_optimization_do:
                        # Define the criterion (loss function)
                        criterion = nn.MSELoss()

                        # Optimize hyperparameters
                        best_parameters, best_score = optimize_neural_network_parameters(
                            X_train, y_train, X_val, y_val, self.optimization_iters, RegressionNN, criterion, target_value_column
                        )

                        # Use the best hyperparameters
                        num_epochs = best_parameters['num_epochs']
                        learning_rate = best_parameters['learning_rate']
                        early_stopping_patience = best_parameters['early_stop_patience']

                        # Initialize the neural network with the best hyperparameters
                        input_dim = X_train.shape[1]
                        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
                        neural_network = RegressionNN(input_dim, output_dim)
                        optimizer = optim.Adam(neural_network.parameters(), lr=learning_rate)
                    else:
                        # Use default parameters and the provided RegressionNN class
                        input_dim = X_train.shape[1]
                        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

                        neural_network = RegressionNN(input_dim, output_dim)
                        optimizer = optim.Adam(neural_network.parameters(), lr=self.learning_rate)
                        num_epochs = self.num_epochs
                        early_stopping_patience = self.early_stopping_patience
                        criterion = nn.MSELoss()

                    # Convert data to PyTorch tensors
                    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
                    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
                    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

                    # Train the neural network
                    train_error, train_error2, val_error, val_error2, y_pred = train_neural_network(
                        neural_network, optimizer, criterion, num_epochs, early_stopping_patience,
                        target_value_column, max(verbose_train, verbose_test),
                        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
                    )

                    init_model = neural_network  # For storage

                store_models(self.result_model_performance, init_model, model,  target_value_column, self.ef_name, train_error, val_error, self.ef_name_2, train_error2, val_error2, X_val, y_val, y_pred)

        if self.save_models:
            self.__save_models()

        return self.result_model_performance

    def __save_models(self):
        for model in self.result_model_performance:
            # Current timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            # Save models with timestamp
            dump(model['model'], f"05_TrainedModels/{self.model_test_name}_{model['target']}_{model['type']}_model_val_rmse_{model['val_RMSE']:.3f}_{timestamp}.joblib")
            print(f"{self.model_test_name}: Saved {model['type']} model for {model['target']} with timestamp {timestamp}")