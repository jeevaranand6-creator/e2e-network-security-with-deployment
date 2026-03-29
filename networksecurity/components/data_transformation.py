import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS, TARGET_COLUMN
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.utils.main_utils.utils import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Extracting imputer parameters from config")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f"Initializing KNNImputer object with config parameters: {DATA_TRANSFORMATION_IMPUTER_PARAMS}")
            pipeline = Pipeline(steps=[("Imputer", imputer)])
            return pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Reading validated train and test file")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Splitting input and target feature from both train and test dataframe")
            x_train, y_train = train_df.drop(TARGET_COLUMN, axis=1), train_df[TARGET_COLUMN]
            x_test, y_test = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y_train = y_train.replace(-1, 0)
            y_test = y_test.replace(-1, 0)
            
            logging.info("Extracting imputer parameters from config")
            preprocessor = self.get_data_transformer_object()

            preprocessor_obj = preprocessor.fit(x_train)
            logging.info("Transforming train and test input features")
            transformed_train = preprocessor_obj.transform(x_train)
            transformed_test = preprocessor_obj.transform(x_test)

            logging.info("Concatenating transformed input features and target feature for train and test data")
            train_arr = np.c_[transformed_train, y_train.to_numpy()]
            test_arr = np.c_[transformed_test, y_test.to_numpy()]

            logging.info("Saving transformed train and test array")
            save_numpy_array_data(file_path=self.data_transformation.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation.transformed_test_file_path, array=test_arr)
            save_object(file_path=self.data_transformation.transformed_object_file_path, obj=preprocessor_obj)

            ## Prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation.transformed_test_file_path
            )

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)