import pandas as pd
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

from typing import List
from datetime import datetime

from modules.utils import load_latest_dataset_from_storage

class ML_Feature_Creator():

    def __init__(self, integrated_data_path: str, threshold: float) -> None:
        """
        ML Feature Creator 
        """
        self.directory = integrated_data_path
        self.threshold = threshold
        self.features_percentiles = None
        self.features_clusters = None

    def run(self):
        print(f'Load Integrated Data from Folder!')
        self.__load_integrated_dataset()
        print('Calculate Percentile Time Features!')
        self.__calc_features_product_change_time_percentiles()
        print('Calculate Time Clusters Features!')
        self.__calc_features_clusters()
        print('Calculate Historic Average Times Features!')
        self.__calc_avg_historic_times_per_product()

        self.__save_features_to_storage()

    def __load_integrated_dataset(self):
        ### Read Data
        data_integrated = load_latest_dataset_from_storage(self.directory, keyword='Fastec_Formate_Dataset')

        # Drop rows with no previous product
        data_integrated = data_integrated.dropna(subset=['Previous_ProductCode'])
        data_integrated = data_integrated[data_integrated['Previous_ProductCode'].str.strip() != '']
        
        # Feature Engineering: Create a unique identifier for each product changeover
        data_integrated['ProductChange'] = data_integrated['Previous_ProductCode'] + "-" + data_integrated['ProductCode']

        # Apply transformation into hours (from milliseconds)
        data_integrated['Auftragswechsel'] = data_integrated['Auftragswechsel']/3600000
        data_integrated['Primär'] = data_integrated['Primär']/3600000
        data_integrated['Sekundär'] = data_integrated['Sekundär']/3600000

        # Save resulting data in global variable
        self.data_integrated = data_integrated

    def __calc_features_product_change_time_percentiles(self, fields=['Auftragswechsel', 'Primär', 'Sekundär']):
        for field in fields:
            # Drop rows with missing combination of product change and product change time time
            data_ml_integration = self.data_integrated.dropna(subset=['ProductChange', field])

            # Apply max time threshold for feature creation
            data_ml_integration.loc[data_ml_integration[field]  > self.threshold, field] = self.threshold

            # Ensure 'ProductChange' is a string
            data_ml_integration['ProductChange'] = data_ml_integration['ProductChange'].astype(str)

            # Extract Percentiles
            # Group by 'OrderCode' and 'ProductChange' and calculate the percentiles for 'Auftragswechsel'
            percentiles_df = data_ml_integration.groupby(['ProductChange'])[field].quantile([0, 0.1, 0.25]).unstack()

            # Rename the columns to reflect the percentile values
            percentiles_df.columns = ['0th_Percentile_'+field, '10th_Percentile_'+field, '25th_Percentile_'+field]

            # Reset the index so 'ProductChange' becomes a column
            percentiles_df.reset_index(inplace=True)

            # Convert 'ProductChange' to string
            percentiles_df['ProductChange'] = percentiles_df['ProductChange'].astype(str)

            if not isinstance(self.features_percentiles, pd.DataFrame):
                self.features_percentiles = percentiles_df
            else:
                self.features_percentiles = self.features_percentiles.merge(percentiles_df, on='ProductChange', how='left')

    def __calc_features_clusters(self, fields=['Auftragswechsel', 'Primär', 'Sekundär']):
        for field in fields:
            # Drop rows with missing combination of product change and product change time time
            data_ml_integration = self.data_integrated.dropna(subset=['ProductChange', field])

            # Apply max time threshold for feature creation
            data_ml_integration.loc[data_ml_integration[field]  > self.threshold, field] = self.threshold

            # Ensure 'ProductChange' is a string
            data_ml_integration['ProductChange'] = data_ml_integration['ProductChange'].astype(str)

            # Extract the relevant features
            clusters_df = data_ml_integration[['ProductChange', field]].groupby('ProductChange', as_index=False)[field].min()

            # Encoding categorical data
            X = pd.get_dummies(clusters_df, columns=['ProductChange'])

            # Gaussian Mixture Model Clustering
            gmm = GaussianMixture(n_components=3, random_state=42)  # Adjust 'n_components' as needed (number of clusters to generate)

            # Save resulting clusters
            clusters_df['Cluster_GMM_'+field] = gmm.fit_predict(X[[field]])

            # Merge Percentile and Cluster Features into result
            clusters_df = clusters_df.drop(columns=[field])

            if not isinstance(self.features_clusters, pd.DataFrame):
                self.features_clusters = clusters_df
            else:
                self.features_clusters = self.features_clusters.merge(clusters_df, on='ProductChange', how='left')

    def __calc_avg_historic_times_per_product(self, fields=['Auftragswechsel', 'Primär', 'Sekundär']):
        self.features_historic_product_times = self.data_integrated[['ProductCode'] + fields]\
                                                    .groupby('ProductCode', as_index=False)[fields]\
                                                    .mean()
        
        # Rename the columns to indicate they are averages
        self.features_historic_product_times.rename(columns={col: f'Historic_Avg_{col}' for col in fields}, inplace=True)

    def __save_features_to_storage(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        
        # Merge Percentiles and Cluster Features in one Dataframe
        self.features_product_change = self.features_clusters.merge(self.features_percentiles, on='ProductChange', how='left')

        self.features_product_change.to_parquet(f'02_FeatureData/ProductChange_Category_Features_{timestamp}.parquet')
        self.features_historic_product_times.to_parquet(f'02_FeatureData/Product_HistoricTimes_Features_{timestamp}.parquet')
        print(f'Stored "02_FeatureData/ProductChange_Category_Features_{timestamp}.parquet', flush=True)
        print(f'Stored "02_FeatureData/Product_HistoricTimes_Features_{timestamp}.parquet', flush=True)