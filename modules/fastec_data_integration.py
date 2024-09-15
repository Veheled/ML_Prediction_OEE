import pandas as pd
import numpy as np

from typing import List

from modules.utils import load_latest_dataset_from_storage

class FastecDataIntegrator():

    def __init__(self, 
                 raw_data_folder: str, 
                 raw_data_file_list: List[str]):
      
      self.raw_data_file_list = raw_data_file_list # Raw Data files to load from the folder
      self.raw_data_folder = raw_data_folder # Folder in which raw data resides

    def run(self) -> pd.DataFrame:
        """
        This function executes the fastec data integration of the data pipeline.

        Returns:
            pd.DataFrame: 
                Fastec Feature Set including all fastec and non fastec defined KPIs and features directly calculated from the fastec raw data.
        """
        self.__load_raw_data()
        print('Loaded Raw Data!', flush=True)
        self.__drop_wrong_columns_from_datasets()
        print('Dropped wrong columns!', flush=True)
        self.__calc_kpis_state_aggregation()
        print('Calculated KPIs based on State Aggregation!', flush=True)
        self.__calc_kpis_counter_aggregation()
        print('Calculated KPIs based on Counter Aggregation!', flush=True)
        self.__calc_kpis_joined_aggregation()
        print('Calculated KPIs based joined Aggregation!', flush=True)
        self.__calc_non_sql_feature_unique_order_information()
        print('Calculated non SQL feature - Unique Order Informations!', flush=True)
        self.__calc_non_sql_feature_previous_product_order()
        print('Calculated non SQL feature - Previous Product / Order!', flush=True)
        self.__calc_non_sql_feature_order_quantity()
        print('Calculated non SQL feature - Order Quantity!', flush=True)
        self.__calc_non_sql_feature_state_durations()
        print('Calculated non SQL features - State Durations!', flush=True)
        self.__calc_non_sql_feature_isMaintenanceOrder()
        print('Calculated non SQL features - Maintenace Order Flag!', flush=True)

        return self.feature_set_fastec
    
    def __load_raw_data(self) -> None:
        """
        This pipeline step loads all the raw data files from the storage folder defined in the initialization of the class.
        """
        for file_keyword in self.raw_data_file_list:
            # Load the latest raw dataset inside the raw data folder
            df = load_latest_dataset_from_storage(
                directory=self.raw_data_folder,
                keyword=file_keyword
            )

            # Dynamically create a class variable with loaded dataframe with name from file_keyword
            setattr(self, file_keyword, df)

    def __drop_wrong_columns_from_datasets(self):
        """
        This pipeline step drops duplicate / faulty / unused and meta data columns from the previously raw data files
        """
        meta_data_columns = ['__InsertDate', '__UpdateDate']
        self.FactMdaState    = self.FactMdaState.drop(columns=meta_data_columns).drop(columns=['MdaStateHashKey'])
        self.FactMdaCounter  = self.FactMdaCounter.drop(columns=meta_data_columns).drop(columns=['MdaCounterHashKey'])
        self.FactMdaMes      = self.FactMdaMes.drop(columns=meta_data_columns)
        self.DimWorkcenter   = self.DimWorkcenter.drop(columns=meta_data_columns).drop(columns=['CostCenter','ExpAvailability','ExpPerformance','ExpQuality','ExpOEE'])
        self.DimShiftEvent   = self.DimShiftEvent.drop(columns=meta_data_columns)
        self.DimMdaOperation = self.DimMdaOperation.drop(columns=meta_data_columns)
        self.DimMdaCounter   = self.DimMdaCounter.drop(columns=meta_data_columns).drop(columns=['MdaCounterId'])
        self.DimMdaState     = self.DimMdaState.drop(columns=meta_data_columns).drop(columns=['MdaStateId'])

    ### Aggregate State based Measures
    def __calc_kpis_state_aggregation(self):
        """
        This pipeline step replicates the Fastec SQL statements related to States + MES integrated KPI calculation.
        """

        merged_df = (
            pd.merge(self.FactMdaState, self.FactMdaMes, left_on='MesHashKey', right_on='MdaMesHashKey', suffixes=('', '_mes'), how='left')
            .merge(self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
            .merge(self.DimMdaState, left_on='StateHashKey', right_on='MdaStateHashKey', suffixes=('', '_std'), how='left')
        )
        
        # Drop Rows that have no state dimension
        merged_df = merged_df.dropna(subset=['Kind'])

        # Pre-calculate conditions
        merged_df['is_planned_stop'] = (merged_df['AvailabilityEnabled'] == True) & (merged_df['Kind'] == 'Planned Stop')
        merged_df['is_unplanned_stop'] = (merged_df['AvailabilityEnabled'] == True) & (merged_df['Kind'] == 'Unplanned Stop')
        merged_df['is_production'] = (merged_df['AvailabilityEnabled'] == True) & (merged_df['Kind'] == 'Production')

        # Pre-calculate other conditions for NOT2
        merged_df['not2_condition'] = (merged_df['PerformanceEnabled'] == False) | \
                                    ((merged_df['ProcessingTime'].isnull() | merged_df['ProcessingTime'] <= 0) & \
                                    (merged_df['WcProcessingTime'].isnull() | merged_df['WcProcessingTime'] <= 0))

        # Group by and aggregate measures
        self.state_aggs = (
            merged_df
            .groupby('MesHashKey')
            .agg(
                RT=('Duration', 'sum'),
                OT=('Duration', lambda x: np.sum(merged_df.loc[x.index, 'is_planned_stop'] * x)),
                DT=('Duration', lambda x: np.sum(merged_df.loc[x.index, 'is_unplanned_stop'] * x)),
                UT=('Duration', lambda x: np.sum(merged_df.loc[x.index, 'is_production'] * x)),
                NOT2=('Duration', lambda x: np.sum(merged_df.loc[x.index, 'not2_condition'] * x))
            )
            .reset_index()
        )

    def __calc_kpis_counter_aggregation(self):
        """
        This pipeline step replicates the Fastec SQL statements related to create counter aggregation based KPIs
        """
        merged_ctf_df = (
            pd.merge(self.FactMdaCounter, self.FactMdaMes, left_on='MesHashKey', right_on='MdaMesHashKey', suffixes=('', '_mes'), how='left')
            .merge(self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
            .merge(self.DimMdaCounter, left_on='CounterHashKey', right_on='MdaCounterHashKey', suffixes=('', '_ctd'), how='left')
        )

        # Convert 'Value' to float to avoid datatype error
        merged_ctf_df['Value'] = merged_ctf_df['Value'].astype(float)

        # Pre-calculate conditions
        merged_ctf_df['is_NOK'] = (merged_ctf_df['QualityEnabled'] == True) & (merged_ctf_df['Kind'] == 'NOK')
        merged_ctf_df['is_Rework'] = (merged_ctf_df['QualityEnabled'] == True) & (merged_ctf_df['Kind'] == 'Rework')
        merged_ctf_df['is_OK'] = (merged_ctf_df['Kind'] == 'OK') | ((merged_ctf_df['QualityEnabled'] == False) & (merged_ctf_df['Kind'] == 'NOK'))

        # Define a custom aggregation function for NOT1
        def calculate_NOT1(x):
            condition1 = (
                (merged_ctf_df.loc[x.index, 'PerformanceEnabled'] == True) & 
                (merged_ctf_df.loc[x.index, 'ProcessingTime'].fillna(merged_ctf_df.loc[x.index, 'WcProcessingTime']).fillna(0) > 0)
            )
            
            condition2 = merged_ctf_df.loc[x.index, 'Kind'].isin(['OK', 'NOK'])
            
            processing_time = merged_ctf_df.loc[x.index, 'ProcessingTime'].fillna(merged_ctf_df.loc[x.index, 'WcProcessingTime']).fillna(0)
            
            return np.sum(
                condition1 * (condition2 * x * processing_time * 1000)
            )

        # Group by and aggregate measures
        self.counter_aggs = (
            merged_ctf_df
            .groupby('MesHashKey')
            .agg(
                SQ=('Value', lambda x: np.sum(merged_ctf_df.loc[x.index, 'is_NOK'] * x)),
                RQ=('Value', lambda x: np.sum(merged_ctf_df.loc[x.index, 'is_Rework'] * x)),
                GQ=('Value', lambda x: np.sum(merged_ctf_df.loc[x.index, 'is_OK'] * x)),
                NOT1=('Value', calculate_NOT1)
            )
            .reset_index()
        )

    def __calc_kpis_joined_aggregation(self):
        """
        This pipeline step replicates the Fastec SQL statements related to creating overall KPIs based on combining the two aggregation views with the shift and dimensional data
        """
        # Join state_aggs and counter_aggs on MesHashKey
        state_counter_kpis = pd.merge(self.state_aggs, self.counter_aggs, on='MesHashKey', how='left')

        # Calculate additional columns
        state_counter_kpis['PBT'] = state_counter_kpis['RT'] - state_counter_kpis['OT']   # PlanOccupancyTime / Planbelegungszeit
        state_counter_kpis['APT'] = state_counter_kpis['PBT'] - state_counter_kpis['DT']  # AllUsageTime / Hauptnutzungszeit
        state_counter_kpis['TQ'] = state_counter_kpis['GQ'] + state_counter_kpis['SQ']    # TotalQuantity / Gesamtmenge

        ### Aggregate KPI's on feature columns
        feature_dim_columns = ['OrderCode']

        # Merge shift information and dimension information with previously aggregated KPIs
        merged_mes_df = (
            pd.merge(self.FactMdaMes, self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
            .merge(self.DimShiftEvent, left_on='ShiftHashKey', right_on='ShiftEventHashKey', suffixes=('', '_sht'), how='left')
            .merge(state_counter_kpis, left_on='MdaMesHashKey', right_on='MesHashKey', how='left')
        )

        # Group by and aggregate
        self.feature_set_fastec = (
            merged_mes_df
            .groupby(feature_dim_columns)
            .agg(
                RT=('RT', 'sum'),
                DT=('DT', 'sum'),
                OT=('OT', 'sum'),
                APT=('APT', 'sum'),
                PBT=('PBT', 'sum'),
                NOT1=('NOT1', 'sum'),
                NOT2=('NOT2', 'sum'),
                GQ=('GQ', 'sum'),
                TQ=('TQ', 'sum'),
                AVAIL=('APT', lambda x: x.sum() / merged_mes_df.loc[x.index, 'PBT'].sum() if merged_mes_df.loc[x.index, 'PBT'].sum() != 0 else 0),
                PERF=('APT', lambda x: (merged_mes_df.loc[x.index, 'NOT1'].sum() + merged_mes_df.loc[x.index, 'NOT2'].sum()) / x.sum() if x.sum() != 0 else 0),
                QUAL=('TQ', lambda x: (merged_mes_df.loc[x.index, 'GQ'].sum() + merged_mes_df.loc[x.index, 'RQ'].sum()) / x.sum() if x.sum() != 0 else 0)
            )
            .reset_index()
        )

        # Calculate OEE
        self.feature_set_fastec['OEE'] = self.feature_set_fastec['AVAIL'] * self.feature_set_fastec['PERF'] * self.feature_set_fastec['QUAL']

    def __calc_non_sql_feature_unique_order_information(self):
        """
        This pipeline step adds order information from the raw data to the integrated dataset like productcode, productgroupcode, and code (=Production Line).
        """
        merged_mes_no_kpi_df = (
            pd.merge(self.FactMdaMes, self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
        )
        unique_order_info = merged_mes_no_kpi_df[['OrderCode', 'ProductCode', 'ProductGroupCode', 'Code']].drop_duplicates().dropna(subset=['OrderCode'])

        self.feature_set_fastec = self.feature_set_fastec.merge(unique_order_info, how='left', on='OrderCode').dropna(subset=['Code'])

    def __calc_non_sql_feature_previous_product_order(self):
        """
        This pipeline step calculates from the raw data the previous order and product for each order and adds it to the feature set.
        """
        # Perform the merges as in your original code
        merged_df = (
            pd.merge(self.FactMdaState, self.FactMdaMes, left_on='MesHashKey', right_on='MdaMesHashKey', suffixes=('', '_mes'), how='left')
            .merge(self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
            .merge(self.DimMdaState, left_on='StateHashKey', right_on='MdaStateHashKey', suffixes=('', '_std'), how='left')
        )

        # Group by 'ProductCode', 'OrderCode', and 'Code', and get the minimum 'Starttime' for each group
        merged_df['Min_Starttime'] = merged_df.groupby(['ProductCode', 'OrderCode', 'Code'])['Starttime'].transform('min')

        # Select only the specified columns and then drop duplicates
        columns_to_extract = ['ProductCode', 'Min_Starttime', 'OrderCode', 'Code']
        unique_rows_df = merged_df[columns_to_extract].drop_duplicates()

        # Now sort by 'Code' and 'Min_Starttime'
        sorted_df = unique_rows_df.sort_values(by=['Code', 'Min_Starttime'], ascending=True).dropna()

        # Group by 'Code' and shift the 'ProductCode' and 'OrderCode' within each group to get the previous values
        sorted_df['Previous_ProductCode'] = sorted_df.groupby('Code')['ProductCode'].shift(1)
        sorted_df['Previous_OrderCode'] = sorted_df.groupby('Code')['OrderCode'].shift(1)

        # Select the columns you are interested in
        columns_to_extract = ['ProductCode', 'Min_Starttime', 'OrderCode', 'Code', 'Previous_ProductCode', 'Previous_OrderCode']

        # Select only the specified columns
        previous_ordercode = sorted_df[columns_to_extract].dropna()

        self.feature_set_fastec = self.feature_set_fastec.merge(previous_ordercode[['OrderCode', 'Previous_ProductCode', 'Previous_OrderCode']], how='left', on='OrderCode')

    def __calc_non_sql_feature_order_quantity(self):
        """
        This pipeline step adds the order quantity to the dataset.
        """
        merged_df = (
            pd.merge(self.FactMdaState, self.FactMdaMes, left_on='MesHashKey', right_on='MdaMesHashKey', suffixes=('', '_mes'), how='left')
            .merge(self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
            .merge(self.DimMdaState, left_on='StateHashKey', right_on='MdaStateHashKey', suffixes=('', '_std'), how='left')
        )
        order_code = merged_df[['OrderCode', 'OrderQuantity']].drop_duplicates()
        duplicate_entries = order_code[order_code['OrderCode'].duplicated(keep=False)]
        filtered_order_quantity = order_code[~order_code['OrderCode'].isin(duplicate_entries['OrderCode'])]

        self.feature_set_fastec = self.feature_set_fastec.merge(filtered_order_quantity, how='left', on='OrderCode')

    def __calc_non_sql_feature_state_durations(self):
        """
        This pipeline adds order information from the raw data to the integrated dataset like productcode, productgroupcode, and code (=Production Line).
        """
        merged_df = (
            pd.merge(self.FactMdaState, self.FactMdaMes, left_on='MesHashKey', right_on='MdaMesHashKey', suffixes=('', '_mes'), how='left')
            .merge(self.DimWorkcenter, left_on='WorkcenterHashKey', right_on='WorkcenterHashKey', suffixes=('', '_wct'), how='left')
            .merge(self.DimMdaOperation, left_on='OperationHashKey', right_on='MdaOperationHashKey', suffixes=('', '_opr'), how='left')
            .merge(self.DimMdaState, left_on='StateHashKey', right_on='MdaStateHashKey', suffixes=('', '_std'), how='left')
        )

        merged_df = merged_df.dropna(subset=['Kind'])

        # Group by 'OrderCode' and 'Category', then sum the 'Duration' for each group
        category_duration_aggs = (
            merged_df
            .groupby(['OrderCode', 'CategoryName'])
            .agg(TotalDuration=('Duration', 'sum'))
            .reset_index()
        )

        # If you want to pivot this for easier readability (each category becomes a column),
        # you can pivot the table after the groupby operation
        category_duration_pivot = category_duration_aggs.pivot(index='OrderCode', columns='CategoryName', values='TotalDuration').reset_index()

        # Fill NaN values with 0 if needed (since not all categories may appear for each 'OrderCode')
        category_duration_pivot = category_duration_pivot.fillna(0)

        self.feature_set_fastec = self.feature_set_fastec.merge(category_duration_pivot, how='left', on='OrderCode')

    def __calc_non_sql_feature_isMaintenanceOrder(self):
        """
        This pipeline step calculates a flag that determines whether or not an order is a maintenance order.
        """
        # Define a function to determine if an order is a maintenance order
        def is_maintenance_order(order_code):
            return str(order_code).startswith('5') and len(str(order_code)) > 6
        
        # Apply the function to the 'OrderCode' column to create a new feature
        self.feature_set_fastec['IsMaintenanceOrder'] = self.feature_set_fastec['OrderCode'].apply(is_maintenance_order).astype(float)