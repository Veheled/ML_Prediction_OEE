import pandas as pd
import numpy as np

from typing import List


class FormateDataIntegrator():

    def __init__(self, formate_file_path: str, sheet_names: List[str], fastec_feature_set: pd.DataFrame) -> None:
        self.formate_file_path = formate_file_path      # Path to formate excel file
        self.sheet_names = sheet_names                  # Excel worksheet names to import
        self.fastec_feature_set = fastec_feature_set    # Fastec Dataintegrated Featureset

    def run(self) -> pd.DataFrame:
        print(f'Load Excel File from {self.formate_file_path}!', flush=True)
        self.__load_formate_excel_sheets()
        print(f'Load "Infos Faltschachtel" from {self.formate_file_path}!', flush=True)
        self.__load_formate_faltschachtel()
        print(f'Load "Infos Packungsbeilage" from {self.formate_file_path}!', flush=True)
        self.__load_formate_packungsbeilage()
        print(f'Load "Infos Tuben" from {self.formate_file_path}!', flush=True)
        self.__load_formate_tuben()

        return self.__feature_transform_product_info(self.fastec_feature_set)

    def __load_formate_excel_sheets(self) -> None:
        ### Read Data
        for sheet_name in self.sheet_names:
            # Load the latest raw dataset inside the raw data folder
            df = pd.read_excel(self.formate_file_path, sheet_name=sheet_name, skiprows=1, dtype=str, engine='openpyxl')

            # Dynamically create a class variable with loaded dataframe with name from file_keyword
            setattr(self, sheet_name+'_infos', df)

    def __load_formate_faltschachtel(self) -> None:
        # Drop unnecessary columns from excel worksheet
        FS_infos_cleaned = self.FS_infos.drop(columns=['Präparatename', 'Lohnauftragg. / Exportpartner', 'V/M', 'Code-Nr.', 'Code-Zeichen', 'Artikel-Nr. in SAP',\
                                                       'Serialisierungspflichtig', 'Aggregation', 'VMF', 'P', 'Linie', 'aktuellste FS-Zeichnung', 'Land',\
                                                        'Darreichungsform', 'neues Logo', 'Stanze', 'Rückstellmuster'])

        FS_infos_cleaned['ProductCode'] = FS_infos_cleaned['Produkt'].astype(str) # Ensure the ProductCode is considered as string not number
        FS_infos_cleaned = FS_infos_cleaned.drop(columns=['Produkt']).dropna().drop_duplicates() # Drop rows with missing values and drop duplicate rows

        print(f"Features Size pre Merge: {len(self.fastec_feature_set)}")
        self.fastec_feature_set = self.fastec_feature_set.merge(FS_infos_cleaned, on='ProductCode', how='left')
        print(f"Feature-Set Size: {len(self.fastec_feature_set)}")

    def __load_formate_packungsbeilage(self) -> None:
        # Drop unnessary columns from excel worksheet
        PBL_infos_cleaned = self.PBL_infos.drop(columns=['Präparatename', 'Code-Nr.', 'Code-Zeichen', 'Artikel-Nr. in SAP', \
                                                         'Linie', 'aktuellste PBL-Zeichnung', 'Darreichungsform', \
                                                         'Land', 'Lohnauftragg. / Exportpartner'])
        # Split rows by linebreaks within the produkt column to get one row per ProductCode
        PBL_infos_cleaned['ProductCode'] = PBL_infos_cleaned['Produkt'].astype(str).str.split('\n')
        PBL_infos_cleaned = PBL_infos_cleaned.explode('ProductCode')

        # Drop rows with missing values and drop duplicate rows
        PBL_infos_cleaned = PBL_infos_cleaned.drop(columns=['Produkt']).dropna().drop_duplicates()

        print(f"Features Size pre Merge: {len(self.fastec_feature_set)}")
        self.fastec_feature_set = self.fastec_feature_set.merge(PBL_infos_cleaned, on='ProductCode', how='left')
        print(f"Feature-Set Size: {len(self.fastec_feature_set)}")

    def __load_formate_tuben(self) -> None:
        # Drop unnessary columns from excel worksheet
        Tuben_infos_cleaned = self.Tuben_infos.drop(columns=['Präparatename', 'Code-Nr.', 'Code-Zeichen', 'Artikel-Nr. in SAP', \
                                                             'V/M' , 'VB', 'Layout', 'Gewinde', 'Land', 'Freifläche'])
        # Split rows by linebreaks within the produkt column to get one row per ProductCode
        Tuben_infos_cleaned['ProductCode'] = Tuben_infos_cleaned['Produkt']
        Tuben_infos_cleaned = Tuben_infos_cleaned.drop(columns=['Produkt'])

        # Drop rows with missing values and drop duplicate rows
        Tuben_infos_cleaned = Tuben_infos_cleaned.dropna().drop_duplicates()

        print(f"Features Size pre Merge: {len(self.fastec_feature_set)}")
        self.fastec_feature_set = self.fastec_feature_set.merge(Tuben_infos_cleaned, on='ProductCode', how='left')
        print(f"Feature-Set Size: {len(self.fastec_feature_set)}")
    
    ## Function to transform product information from strings into multiple integer features
    # Requires the input to be a dataframe that contains the columns used below
    def __feature_transform_product_info(self, features: pd.DataFrame) -> pd.DataFrame:
        # Splitting FS-Größe into FS Breite, Länge, Tiefe
        features[['FS_Breite', 'FS_Länge', 'FS_Tiefe']] = features['FS-Größe'].str.extract(r'(\d+)x(\d+)x(\d+)').astype(float)
        
        # Splitting PBL-Größe into PBL Länge, Breite
        features[['PBL_Länge', 'PBL_Breite']] = features['PBL-Größe'].str.extract(r'(\d+)x(\d+)').astype(float)
        
        # Splitting Tuben-Größe into Tuben Durchmesser, Länge
        # Handling the comma as decimal separator and Ø character
        features[['Tuben_Durchmesser', 'Tuben_Länge']] = features['Tuben-Größe'].str.replace('Ø', '').str.replace(',', '.').str.extract(r'(\d+\.?\d*)x(\d+)').astype(float)
        
        features = features.drop(columns=['FS-Größe', 'PBL-Größe', 'Tuben-Größe'])
        return features
