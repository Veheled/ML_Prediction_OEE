# Project Repository Overview

This repository contains Jupyter notebooks and Python modules designed for data processing, feature engineering, model training, and deployment through a web interface. The repository is structured to guide the user from loading raw data to optimizing a machine learning model, and includes integration with a simple Flask-based web interface.

## Notebooks Overview

The notebooks follow a logical pipeline for machine learning model development:

1. **00_LoadRawData.ipynb**  
   - Loads and preprocesses raw data from multiple sources for further analysis and model training.
   
2. **01_DataIntegration.ipynb**  
   - Integrates data from different datasets, ensuring consistency and completeness for downstream tasks.
   
3. **02_FeatureCreation.ipynb**  
   - Performs feature engineering on the integrated data to create meaningful features for the machine learning model.

4. **03_ModelTraining.ipynb**  
   - Trains machine learning models using the processed and feature-engineered dataset.

5. **04_FrontendSetup.ipynb**  
   - Sets up the web frontend using Flask, which allows interaction with the trained model via a web interface.

6. **05_ModelEvaluation.ipynb**  
   - Evaluates the performance of the trained model using various metrics to ensure it meets the desired accuracy and generalization.

7. **06_ModelOptimization.ipynb**  
   - Optimizes the model hyperparameters and architecture for better performance.

## Python Modules Overview

The Python modules provide utility functions and custom classes to handle different stages of the project:

1. **dataloader.py**  
   - Contains functions to load raw data from files and databases, handling various file formats and ensuring data consistency.

2. **fastec_data_integration.py**  
   - Provides functionality to integrate and preprocess FASTEC-specific data sources into the pipeline.

3. **feature_creator.py**  
   - Implements feature engineering steps such as feature transformations, generation of new features, and handling categorical and numerical variables.

4. **flask_webinterface.py**  
   - Contains code to set up a Flask-based web interface, allowing users to interact with the trained model via a simple UI.

5. **formate_data_integration.py**  
   - Focuses on integrating data from the Formate dataset, preparing it for feature engineering and model training.

6. **model_training.py**  
   - Includes functions and classes to train machine learning models, handle different algorithms, and manage training configurations.

7. **rüstmatrix_data_integrator.py**  
   - Integrates and processes data from the Rüstmatrix system, ensuring compatibility with the rest of the data pipeline.

8. **training_pipeline_steps.py**  
   - Defines the key steps in the training pipeline, such as data preprocessing, model training, and evaluation.

9. **utils.py**  
   - Contains utility functions used across the repository, including helper methods for data manipulation, file I/O, and logging.

## Getting Started

### Requirements

- Python 3.x
- Jupyter Notebook
- Flask
- Pandas, Scikit-learn, and other machine learning libraries

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name

2. Install dependencies:
   pip install -r requirements.txt

3. Run Jupyter notebooks to follow the pipeline from raw data loading to model training and deployment.

### Running the Web Interface
To interact with the trained model via the web interface:

1. Navigate to the 04_FrontendSetup.ipynb.
2. Run the Notebook within VSCode or similar
3. Open your browser and go to http://localhost:5000 to interact with the model.

### License
This project is licensed under the MIT License.
