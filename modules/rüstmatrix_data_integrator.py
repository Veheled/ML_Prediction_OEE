import pandas as pd
import numpy as np

from typing import List

from modules.utils import load_latest_dataset_from_storage

class RüstmatrixDataIntegrator():

    def __init__(self, formate_file_path: str, rüstmatrix_file_path: str, join_feature_set: pd.DataFrame) -> None:
        self.formate_file_path = formate_file_path              # Path to formate excel file
        self.rüstmatrix_file_path = rüstmatrix_file_path        # Path to rüstmatrix excel file
        self.join_feature_set = join_feature_set                # Featureset to join the results into

    def run(self) -> pd.DataFrame:
        print(f'Load Excel File Fromate from {self.formate_file_path}!', flush=True)
        self.__load_excel_sheets(self.formate_file_path, ['FS'], 1)
        print(f'Load Excel File Rüstmatrix from {self.rüstmatrix_file_path}!', flush=True)
        self.__load_excel_sheets(self.rüstmatrix_file_path, ['RM'], 0)
        print(f'Load "Infos Faltschachtel" from {self.formate_file_path}!', flush=True)
        self.__calculate_rüstmatrix()
        print(f'Calculate Rüstmatrix Information!', flush=True)

        return self.join_feature_set.merge(self.final_RM_infos, how='left', on='ProductCode')

    def __load_excel_sheets(self, file_path, sheet_names, skiprows) -> None:
        ### Read Data
        for sheet_name in sheet_names:
            # Load the latest raw dataset inside the raw data folder
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=skiprows, dtype=str, engine='openpyxl')

            # Dynamically create a class variable with loaded dataframe with name from file_keyword
            setattr(self, sheet_name+'_infos', df)

    def __calculate_rüstmatrix(self) -> None:
        Transform = self.FS_infos[['Produkt', 'Darreichungsform']].dropna().drop_duplicates()
        Transform['Darreichungsform_Num'] = Transform['Darreichungsform'].str.extract('(\d+)').astype(int)
        Transform['ProductCode'] = Transform['Produkt']
        Transform = Transform[['ProductCode', 'Darreichungsform_Num']]

        self.RM_infos['ProductCode'] = self.RM_infos['MATNR'].str.lstrip('0')
        self.final_RM_infos = Transform.merge(self.RM_infos[['ProductCode', 'ZZ_PACKGROESSE', 'ZZPCKGR', 'ZALUFOL', 'ZWIRKST', 'ZZWRKST']], how='left', on='ProductCode')

        # Create the new columns based on the provided logic
        self.final_RM_infos['CALC_PACKGROESSE'] = self.final_RM_infos['ZZPCKGR'].fillna(self.final_RM_infos['ZZ_PACKGROESSE']).fillna(self.final_RM_infos['Darreichungsform_Num']).astype(float)
        self.final_RM_infos['CALC_WIRKSTOFF'] = self.final_RM_infos['ZWIRKST'].astype(str)#.fillna(final_RM_infos['ZZWRKST'])
        self.final_RM_infos['CALC_ALUFOLIE'] = self.final_RM_infos['ZALUFOL'].astype(str)#.fillna(final_RM_infos['ZZALUFOL'])
        self.final_RM_infos = self.final_RM_infos[['ProductCode','CALC_PACKGROESSE', 'CALC_WIRKSTOFF', 'CALC_ALUFOLIE']]