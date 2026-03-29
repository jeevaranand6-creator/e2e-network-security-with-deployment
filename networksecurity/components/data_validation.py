from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os, sys

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, 
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    
    def validate_no_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            no_of_columns = len(self._schema_config)
            logging.info(f"Required no of columns: {no_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            return len(dataframe.columns) == no_of_columns
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def detect_dataset_drift(self, base_df, current_df, threshold=0.5) -> bool:
        try:
            report = {}
            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_sample_dist=ks_2samp(d1, d2)
                is_found = threshold<=is_sample_dist.pvalue
                status = not is_found
                report.update({
                    column: {
                        "p_value": float(is_sample_dist.pvalue),
                        "drift_status": is_found
                    }
                })
                drift_report_file_path = self.data_validation_config.drift_report_file_path
                
                #create directory
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path, exist_ok=True)
                write_yaml_file(file_path=drift_report_file_path, content=report)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            status = self.validate_no_of_columns(dataframe=train_df)
            if not status:
                error_message = f"Train data frame does not contain all columns.\n"
            status = self.validate_no_of_columns(dataframe=test_df)
            if not status:
                error_message = f"Test data frame does not contain all columns.\n"

            ##lets check datadrift
            self.detect_dataset_drift(base_df=train_df, current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)
            train_df.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True
            )

            test_df.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )

            datavalidationartifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_test_file_path=None,
                invalid_train_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            return datavalidationartifact
        
        except Exception as e:
            raise NetworkSecurityException(e, sys)