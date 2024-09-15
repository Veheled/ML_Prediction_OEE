import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

# Import sklearn support methods
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator

# Import model types
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import Pytorch Neural Network
import torch
import torch.nn as nn
import torch.optim as optim

# Import Optuna for hyperparameter optimization:
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from optuna.integration import OptunaSearchCV
#optuna.logging.set_verbosity(optuna.logging.ERROR)  # Only show errors

from scipy.stats import randint as sp_randint
from scipy.stats import uniform
import time

# Define the neural network
class RegressionNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, output_dim)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

## Function to extract feature columns from complete dataset and separate target value from feature set
def feature_column_selection(raw_data, feature_columns, target_value_columns):
    ### Select features and target and apply filters on faulty data  
    features = raw_data.copy()
    features = features[feature_columns + target_value_columns]

    # Apply thresholds and proper scaling for target values
    for target in target_value_columns:
        if target in ['OEE', 'PERF', 'AVAIL', 'QUAL']:
            features[target] = features[target].clip(0, 1.25)  # Clip target values between 0 and 1.25
        elif target in ['RT', 'DT', 'OT', 'APT', 'PBT', 'NOT1', 'NOT2']:
            features[target] = (features[target] / 3600000).clip(0, None)

    # Drop orders which are missing crucial feature data
    features = features.dropna(subset=['Previous_ProductCode'])
    features = features.dropna(subset=['OrderQuantity'])

    return features[feature_columns], features[target_value_columns].astype('float64')  # Ensure float64 dtype

# Standard Normalize Features for models that require this
def feature_numeric_normalization(features, do_norm):
    features_norm = features.copy()

    if do_norm:
        # Automatically select numeric columns to normalize
        numeric_columns = features.select_dtypes(include=['float64', 'int64', 'int32']).columns
        
        for col in numeric_columns:
            # Fit and transform the column with the scaler
            scaler = StandardScaler()
            # Fill NaNs with 0 before scaling
            features_norm[col].fillna(0, inplace=True)
            features_norm[col] = scaler.fit_transform(features_norm[col].values.reshape(-1, 1))

    else:
        # If normalization is not needed, just fill NaNs with 0
        features_norm.fillna(0, inplace=True)
    
    return features_norm

# Function to encode categorical features and ensure all columns are label
def feature_encoding(features, drop, to_be_encoded, encoding_type='ordinal'):
    new_columns = []  # List to keep track of new column names
    
    # Encode all columns except 'Code' with either LabelEncoder or OrdinalEncoder
    for column in to_be_encoded:
        if column != 'Code':  # Skip the 'Code' column here
            # Fill missing values with 'Missing'
            features[column].fillna('Missing', inplace=True)
            
            new_col_name = column + '_encoded'
            
            # Switch between LabelEncoder and OrdinalEncoder based on input
            if encoding_type == 'label':
                label_encoder = LabelEncoder()
                features[new_col_name] = label_encoder.fit_transform(features[column]).astype('float64')
            elif encoding_type == 'ordinal':
                ordinal_encoder = OrdinalEncoder()
                features[new_col_name] = ordinal_encoder.fit_transform(features[[column]]).astype('float64')
            
            new_columns.append(new_col_name)

    # Apply One-Hot Encoding to 'Code' column
    if 'Code' in features.columns:
        one_hot_encoded = pd.get_dummies(features['Code'], prefix='Code', dtype='float64')
        features = pd.concat([features, one_hot_encoded], axis=1)
        new_columns.extend(one_hot_encoded.columns.tolist())  # Add new one-hot columns to the tracking list

    if drop:
        # Drop original encoded columns
        features = features.drop(columns=to_be_encoded + ['Code'])

        # Reorder DataFrame to have the new columns at the beginning
        columns_order = new_columns + [col for col in features.columns if col not in new_columns]
        features = features[columns_order]

    return features

# Function to initialize target models and return them to the pipeline
def initialize_model(model, best_parameters):
    # Dictionary of models and their corresponding constructors
    model_switch = {
        'linear': LinearRegression,
        'ridge': Ridge,
        'poly': PolynomialFeatures,
        'dt': DecisionTreeRegressor,
        'rf': RandomForestRegressor,
        'xgb': XGBRegressor,
        'svr': SVR,
        'lgbm': LGBMRegressor,
        'catboost': CatBoostRegressor
    }

    # Get the model constructor
    model_constructor = model_switch.get(model)

    if not model_constructor:
        return None

    # If the model is Polynomial Regression (poly), create a pipeline with PolynomialFeatures and LinearRegression
    if model == 'poly':
        if best_parameters:
            degree = best_parameters.get('degree', 2)  # Default degree is 2 if not provided
        else:
            degree = 2
        return make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Get the valid parameters for the model constructor
    valid_params = model_constructor().get_params().keys()

    # Handle default model initialization with no parameters
    if best_parameters is None:
        if model == 'catboost':
            return model_constructor(silent=True)  # CatBoost: Completely disable logging
        elif model == 'lgbm':
            return model_constructor(verbose=-1)  # LightGBM: Completely disable logging
        elif 'verbose' in model_constructor().get_params().keys():
            return model_constructor(verbose=False)
        else:
            return model_constructor()

    # Initialize model with best parameters from optimization
    if model == 'catboost':
        return model_constructor(silent=True, **best_parameters)  # Silent mode for CatBoost
    elif model == 'lgbm':
        return model_constructor(verbose=-1, **best_parameters)  # Silent mode for LightGBM
    elif 'verbose' in model_constructor().get_params().keys():
        return model_constructor(verbose=False, **best_parameters)
    else:
        return model_constructor(**best_parameters)

    
## Function to get model grid search parameter grid 
def get_model_param_grid(model):
    param_grid_switch = {
        'linear': {},  # No hyperparameters for basic Linear Regression
        'ridge': {
            'alpha': [0.01, 0.1, 1, 10, 100]
        },
        'poly': {
            'poly__degree': [2, 3, 4]
        },
        'dt': {
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'rf': {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        'xgb': {
            'n_estimators': [100, 200, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.5]
        },
        'svr': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.01, 0.1, 0.5]
        },
        'lgbm': {
            'n_estimators': [50, 100, 200, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves': [31, 62, 127, 255],
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1],
            'reg_lambda': [0, 0.1, 0.5, 1]
        },
        'catboost': {
            'iterations': [100, 300, 500],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [4, 6, 8, 10],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128, 255],
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
        }
    }
    return param_grid_switch.get(model, None)

## Function to get model randomized hyperparameter search parameter grid 
def get_model_random_search_params(model):
    param_dist_switch = {
        'linear': {},  # No hyperparameters for basic Linear Regression
        'ridge': {
            'alpha': uniform(0.01, 100)
        },
        'poly': {
            'poly__degree': sp_randint(2, 5)
        },
        'dt': {
            'max_depth': sp_randint(10, 40),
            'min_samples_split': sp_randint(2, 11),
            'min_samples_leaf': sp_randint(1, 5)
        },
        'rf': {
            'n_estimators': sp_randint(50, 500),
            'max_depth': [None] + list(range(10, 50, 10)),
            'min_samples_split': sp_randint(2, 11),
            'min_samples_leaf': sp_randint(1, 5),
            'bootstrap': [True, False]
        },
        'xgb': {
            'n_estimators': sp_randint(50, 500),
            'max_depth': sp_randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'gamma': uniform(0, 0.5)
        },
        'svr': {
            'C': uniform(0.1, 100),
            'gamma': ['scale', 'auto'] + list(uniform(0.01, 1).rvs(10)),
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'epsilon': uniform(0.01, 0.5)
        },
        'lgbm': {
            'n_estimators': sp_randint(50, 500),
            'learning_rate': uniform(0.01, 0.2),
            'num_leaves': sp_randint(31, 255),
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'colsample_bytree': uniform(0.6, 0.4),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 1)
        },
        'catboost': {
            'iterations': sp_randint(100, 500),
            'learning_rate': uniform(0.01, 0.2),
            'depth': sp_randint(4, 10),
            'l2_leaf_reg': sp_randint(1, 9),
            'border_count': sp_randint(32, 255),
            'grow_policy': ['SymmetricTree', 'Depthwise', 'Lossguide']
        }
    }
    return param_dist_switch.get(model, None)

from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

# Function to get model parameter distribution for Optuna optimization
def get_model_param_optuna(model):
    param_dist_switch = {
        'rf': {
            'n_estimators': IntDistribution(50, 500),
            'max_depth': IntDistribution(10, 50),
            'min_samples_split': IntDistribution(2, 11),
            'min_samples_leaf': IntDistribution(1, 5),
            'bootstrap': CategoricalDistribution([True, False])
        },
        'xgb': {
            'n_estimators': IntDistribution(50, 500),
            'max_depth': IntDistribution(3, 10),
            'learning_rate': FloatDistribution(0.01, 0.2),
            'subsample': FloatDistribution(0.6, 1.0),
            'colsample_bytree': FloatDistribution(0.6, 1.0),
            'gamma': FloatDistribution(0, 0.5)
        },
        'svr': {
            'C': FloatDistribution(0.1, 10, log=True),
            'epsilon': FloatDistribution(0.01, 0.5, log=True),
            'kernel': CategoricalDistribution(['rbf', 'linear'])
            # 'gamma' will be conditionally included in the objective function
        },
        'lgbm': {
            'n_estimators': IntDistribution(50, 500),
            'learning_rate': FloatDistribution(0.01, 0.2),
            'num_leaves': IntDistribution(31, 255),
            'boosting_type': CategoricalDistribution(['gbdt', 'dart', 'goss']),
            'colsample_bytree': FloatDistribution(0.6, 1.0),
            'reg_alpha': FloatDistribution(0, 1),
            'reg_lambda': FloatDistribution(0, 1)
        },
        'catboost': {
            'iterations': IntDistribution(100, 500),
            'learning_rate': FloatDistribution(0.01, 0.2),
            'depth': IntDistribution(4, 10),
            'l2_leaf_reg': IntDistribution(1, 9),
            'border_count': IntDistribution(32, 255),
            'grow_policy': CategoricalDistribution(['SymmetricTree', 'Depthwise', 'Lossguide'])
        },
        'ridge': {
            'alpha': FloatDistribution(0.01, 100)
        },
        'dt': {
            'max_depth': IntDistribution(10, 40),
            'min_samples_split': IntDistribution(2, 11),
            'min_samples_leaf': IntDistribution(1, 5)
        },
        'linear': {
            # Linear regression does not have hyperparameters to tune
        },
        'poly': {
            # Polynomial regression - tune the degree of the polynomial
            'poly__degree': IntDistribution(2, 4)
        }
    }
    return param_dist_switch.get(model, None)

def optimize_model_parameters(optimization_mode, model, model_name, X_gridsearch, y_gridsearch, n_iter_search):
    param_grid = None

    if model_name == 'linear':
        # Linear regression does not need hyperparameter tuning
        return None, None
    else:
        # Define KFold with the same configuration as in your training function
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        if model_name == 'poly':
            # Create a pipeline with PolynomialFeatures and LinearRegression
            pipeline = Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LinearRegression())
            ])

            #if optimization_mode == 'gridsearch':
            param_grid = get_model_param_grid(model_name)
            search_model = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=kf,
                scoring='neg_root_mean_squared_error',
                verbose=0,
                n_jobs=-1
            )

            #elif optimization_mode == 'randomsearch':
            #    param_dist = get_model_random_search_params(model_name)
            #    search_model = RandomizedSearchCV(
            #        pipeline,
            #        param_distributions=param_dist,
            #        n_iter=6,
            #        cv=kf,
            #        scoring='neg_root_mean_squared_error',
            #        verbose=0,
            #        n_jobs=-1,
            #        random_state=42
            #    )
#
            #elif optimization_mode == 'optunasearch':
            #    param_dist = get_model_param_optuna(model_name)
            #    search_model = OptunaSearchCV(
            #        pipeline,
            #        param_distributions=param_dist,
            #        cv=kf,
            #        n_trials=6,
            #        scoring='neg_root_mean_squared_error',
            #        verbose=0,
            #        n_jobs=-1,
            #        random_state=42
            #    )

            # Fit the search model
            search_model.fit(X_gridsearch, y_gridsearch)

            # Print the best parameters and scores
            print(f"Best parameters for {model_name} with {optimization_mode} are: {search_model.best_params_}")
            print(f"Best score for {model_name} with {optimization_mode} is: {-search_model.best_score_:.3f}")

            return search_model.best_params_, -search_model.best_score_

        else:
            if optimization_mode == 'gridsearch':
                param_grid = get_model_param_grid(model_name)
                search_model = GridSearchCV(
                    model,
                    param_grid,
                    cv=kf,
                    scoring='neg_root_mean_squared_error',
                    verbose=0,
                    n_jobs=-1
                )

            elif optimization_mode == 'randomsearch':
                param_grid = get_model_random_search_params(model_name)
                search_model = RandomizedSearchCV(
                    model,
                    param_grid,
                    n_iter=n_iter_search,
                    cv=kf,
                    scoring='neg_root_mean_squared_error',
                    verbose=0,
                    n_jobs=-1,
                    random_state=42
                )

            elif optimization_mode == 'optunasearch':
                param_grid = get_model_param_optuna(model_name)
                search_model = OptunaSearchCV(
                    model,
                    param_distributions=param_grid,
                    cv=kf,
                    n_trials=n_iter_search,
                    scoring='neg_root_mean_squared_error',
                    verbose=0,
                    n_jobs=-1,
                    random_state=42
                )

            # Fit the search model
            search_model.fit(X_gridsearch, y_gridsearch)

            # Print the best parameters and scores
            print(f"Best parameters for {model_name} with {optimization_mode} are: {search_model.best_params_}")
            print(f"Best score for {model_name} with {optimization_mode} is: {-search_model.best_score_:.3f}")

            return search_model.best_params_, -search_model.best_score_


def optimize_neural_network_parameters(X_train, y_train, X_val, y_val, n_iter_search, base_model, criterion, target_value_column):
    def objective(trial):
        # Suggest hyperparameters
        num_epochs = trial.suggest_int('num_epochs', 100, 2000)
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
        early_stopping_patience = trial.suggest_int('early_stop_patience', 25, 500)

        # Initialize the neural network with the base model
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
        model = base_model(input_dim, output_dim)

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

        # Train the neural network
        train_score, train_score2, val_score, val_score2, _ = train_neural_network(
            model, optimizer, criterion, num_epochs, early_stopping_patience,
            target_value_column=target_value_column,
            verbose=0,  # Suppress outputs during optimization
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            X_val_tensor=X_val_tensor,
            y_val_tensor=y_val_tensor
        )

        # Return validation RMSE as the objective to minimize
        return val_score

    # Create and run the Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_iter_search)

    best_parameters = study.best_params
    best_score = study.best_value

    print(f"Best parameters for NN with OptunaSearch are: {best_parameters}")
    print(f"Best score for NN with OptunaSearch is: {best_score:.3f}")

    return best_parameters, best_score

## Function to plot prediction chart
def pred_act_plot(all_actuals, all_predictions, target_value_column, model_name, plot_mode):
    import matplotlib.pyplot as plt
    import scienceplots

    # Use the 'science' style for plots
    plt.style.use('science')

    # Plotting Actual vs Predicted values
    plt.figure(figsize=(5, 5))  # Making the figure square
    plt.scatter(all_actuals, all_predictions, alpha=0.5)
    plt.title(f'{target_value_column} - {model_name}: Actual vs Predicted - {plot_mode} Dataset')
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.plot([0, 125], [0, 125], 'k--', lw=2)
    if target_value_column in ['OEE', 'AVAIL', 'PERF', 'QUAL']:
        plt.xlim(all_predictions.min(), all_predictions.max())  # Limiting X-axis
        plt.ylim(all_predictions.min(), all_predictions.max())  # Limiting Y-axis
    plt.gca().set_aspect('equal', adjustable='box')  # Ensuring the plot is quadratic
    plt.show()
    print("~~~~~~~~~~~~")

## Function to train models via kFold cross validation
def train_model_kfold_cross_val(model, model_name, target_value_column, n_splits, X, y, ef_name, score_function, ef_name2, score_function2, verbose):
    # To store scores
    scores = []
    scores2 = []
    scores3 = []
    all_predictions = []
    all_actuals = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index].values.ravel(), y.iloc[test_index].values.ravel()  # Use ravel() to ensure 1D array

        model.fit(X_train, y_train)  # Fit model to train data slice

        if target_value_column in ['OEE', 'PERF', 'AVAIL', 'QUAL']:
            predictions_train = model.predict(X_train).clip(0, 1.25)  # Predict train data slice
            predictions_test = model.predict(X_test).clip(0, 1.25)  # Predict test data slice
        else:
            predictions_train = model.predict(X_train).clip(0, None)  # Predict train data slice
            predictions_test = model.predict(X_test).clip(0, None)  # Predict test data slice

        score = score_function(y_train, predictions_train)  # Calculate scores over train slice
        score2 = score_function2(y_train, predictions_train)  # Calculate second informative error function
        score3 = r2_score(y_train, predictions_train)  # Calculate second informative error function

        if ef_name == 'RMSE':
            score = math.sqrt(score)  # If RMSE, calculate sqrt of scores

        # Store result in lists
        scores.append(score)
        scores2.append(score2)
        scores3.append(score3)
        all_predictions.extend(predictions_test)
        all_actuals.extend(y_test)
  
    # Calculate errors over all folds combined
    combined_score1 = math.sqrt(score_function(all_actuals, all_predictions))
    combined_score2 = score_function2(all_actuals, all_predictions)
    combined_score3 = r2_score(all_actuals, all_predictions)

    if verbose == 2:
        print(f"Target: {target_value_column} --> {model_name} - Training Average {ef_name} per fold: {np.mean(scores):.3f} - average {ef_name2} {np.mean(scores2):.3f}")
        print(f"Target: {target_value_column} --> {model_name} - {ef_name} over all folds: {combined_score1:.3f} - {ef_name2} over all folds: {combined_score2:.3f} - R2 over all folds: {combined_score3:.3f}")
        pred_act_plot(all_actuals, all_predictions, target_value_column, model_name, "All Folds - Training")
    elif verbose == 1:
        print(f"Target: {target_value_column} --> {model_name} - Training Average {ef_name} per fold: {np.mean(scores):.3f} - average {ef_name2} {np.mean(scores2):.3f}")
        print(f"Target: {target_value_column} --> {model_name} - {ef_name} over all folds: {combined_score1:.3f} - {ef_name2} over all folds: {combined_score2:.3f} - R2 over all folds: {combined_score3:.3f}")
    ##else:
        ## No output

    return combined_score1, combined_score2, scores, all_actuals, all_predictions

def train_neural_network(model, optimizer, criterion, num_epochs, early_stopping_patience, target_value_column, verbose, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    # Training the model
    scores = []
    all_predictions = []
    all_actuals = []

    # Lists to store losses
    train_losses = []
    val_losses = []

    # Early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        
        # Clip predictions in training based on target column
        if target_value_column in ['OEE', 'PERF', 'AVAIL', 'QUAL']:
            outputs = torch.clamp(outputs, min=0, max=1.25)  # Clip between 0 and 1.25
        else:
            outputs = torch.clamp(outputs, min=0)  # Clip to non-negative
        
        loss = torch.sqrt(criterion(outputs, y_train_tensor))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            
            # Clip validation predictions
            if target_value_column in ['OEE', 'PERF', 'AVAIL', 'QUAL']:
                val_outputs = torch.clamp(val_outputs, min=0, max=1.25)  # Clip between 0 and 1.25
            else:
                val_outputs = torch.clamp(val_outputs, min=0)  # Clip to non-negative
            
            val_loss = torch.sqrt(criterion(val_outputs, y_val_tensor))
            val_losses.append(val_loss.item())

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if verbose >= 1:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        if (epoch+1) % 50 == 0 and verbose == 2:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train RMSE: {loss.item():.4f}, Validation RMSE: {val_loss.item():.4f}')

    # Collect predictions and actuals after training
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_tensor).numpy().flatten()
        y_train_numpy = y_train_tensor.numpy().flatten()
        val_outputs = model(X_val_tensor).numpy().flatten()
        y_val_numpy = y_val_tensor.numpy().flatten()

        # Clip predictions after collecting them
        if target_value_column in ['OEE', 'PERF', 'AVAIL', 'QUAL']:
            train_outputs = np.clip(train_outputs, 0, 1.25)  # Clip between 0 and 1.25
            val_outputs = np.clip(val_outputs, 0, 1.25)  # Clip between 0 and 1.25
        else:
            train_outputs = np.clip(train_outputs, 0, None)  # Clip to non-negative
            val_outputs = np.clip(val_outputs, 0, None)  # Clip to non-negative

    # Calculate score for validation set
    train_score = np.sqrt(mean_squared_error(train_outputs, y_train_numpy))
    train_score2 = mean_absolute_error(train_outputs, y_train_numpy)
    val_score = np.sqrt(mean_squared_error(val_outputs, y_val_numpy))
    val_score2 = mean_absolute_error(val_outputs, y_val_numpy)
    val_score3 = r2_score(val_outputs, y_val_numpy)
    all_predictions = val_outputs
    all_actuals = y_val_numpy

    if verbose == 2:
        # Plotting training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train RMSE')
        plt.plot(val_losses, label='Validation RMSE')
        plt.xlabel('Epochs')
        plt.ylabel('RMSE')
        plt.legend()
        plt.title(f'{target_value_column} - Training and Validation RMSE over Epochs')
        plt.show()

        print(f"Target: {target_value_column} --> Neural Network - Final Validation RMSE : {val_score:.3f} - MAE: {val_score2:.3f} - R2: {val_score3:.3f}\n~~~~~~~~~~~~")
        # Plotting actual vs predicted for training
        pred_act_plot(y_train_numpy, train_outputs, target_value_column, "Neural Network", "Validation")

    if verbose >= 1:
        print(f"Target: {target_value_column} --> Neural Network - Final Validation RMSE : {val_score:.3f} - MAE: {val_score2:.3f} - R2: {val_score3:.3f}\n~~~~~~~~~~~~")
        
    return train_score, train_score2, val_score, val_score2, all_predictions

## Function to extract validation error for given model over given validation dataset and error functions
def validate_model(model, model_name, target_value_column, X_test, y_test, ef_name, error_function, ef_name_2, ef_2, verbose):

    if target_value_column in ['OEE', 'PERF', 'AVAIL', 'QUAL']:
        all_predictions = model.predict(X_test).clip(0, 1.25)  # Predict test data slice
    else:
        all_predictions = model.predict(X_test).clip(0, None)  # Predict test data slice    
    
    score = error_function(y_test, all_predictions)

    if ef_name =='RMSE':
        score = math.sqrt(score)
    score2 = ef_2(y_test, all_predictions)
    score3 = r2_score(y_test, all_predictions)
    all_actuals = y_test
    
    if verbose == 2:
        print(f"Target: {target_value_column} --> {model_name} Validation - {ef_name}={score:.3f}, {ef_name_2}={score2:.3f}, R2={score3:.3f}")
        pred_act_plot(all_actuals, all_predictions, target_value_column, model_name, "Validation")
    elif verbose == 1:
        print(f"Target: {target_value_column} --> {model_name} Validation - {ef_name}={score:.3f}, {ef_name_2}={score2:.3f}, R2={score3:.3f}\n~~~~~~~~~~~~")
    ##else:
        ## No output

    return score, score2, all_predictions

## Function to validate composite model of OEE from subcomponents
def validate_model_composite(predictions, model_name, target_value_column, y_test, ef_name, error_function, ef_name_2, ef_2, verbose):

    all_predictions = predictions
    score = error_function(y_test, all_predictions)
    if ef_name =='RMSE':
        score = math.sqrt(score)
    score2 = ef_2(y_test, all_predictions)
    all_actuals = y_test
    
    if verbose == 2:
        print(f"Target: {target_value_column} --> {model_name} Validation - {ef_name}={score:.3f}, {ef_name_2}={score2:.3f}")
        pred_act_plot(all_actuals, all_predictions, target_value_column, model_name, "Validation")
    elif verbose == 1:
        print(f"Target: {target_value_column} --> {model_name} Validation - {ef_name}={score:.3f}, {ef_name_2}={score2:.3f}\n~~~~~~~~~~~~")
    ##else:
        ## No output

    return score, score2, all_predictions

## Store models in list for storage and further analysis
def store_models(result_model_performance, model, type, target_value_column, ef_name, train_error, val_error, ef_name2, train_error2, val_error2, X_val, y_val, y_pred):
    result_model_performance.append({
        'model': model, 
        'type': type,
        'target': target_value_column,
        'train_'+ef_name: train_error,
        'train_'+ef_name2: train_error2,
        'val_'+ef_name: val_error,
        'val_'+ef_name2: val_error2,
        'X_val': X_val,
        'y_val': y_val,
        'y_pred': y_pred
    })