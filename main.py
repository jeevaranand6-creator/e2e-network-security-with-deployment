from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig, DataTransformationConfig, ModelTrainerConfig, TrainingPipelineConfig, DataValidationConfig
from networksecurity.components.data_validation import DataValidation
import sys

if __name__ == "__main__":

    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(data_ingestion_config=dataingestionconfig)
        logging.info("Initiate data ingestion")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data Initiation completed")
        print(dataingestionartifact)
        
        datavalidationconfig = DataValidationConfig(training_pipeline_config=trainingpipelineconfig)
        datavalidation = DataValidation(dataingestionartifact, datavalidationconfig)
        logging.info("Initiate data validation")
        datavalidationartifact = datavalidation.initiate_data_validation()
        logging.info("Data validation completed")
        print(datavalidationartifact)
        
        datatransformationconfig = DataTransformationConfig(training_pipeline_config=trainingpipelineconfig)
        datatransformation = DataTransformation(datavalidationartifact, datatransformationconfig)
        logging.info("Initiate data transformation")
        datatransformationartifact = datatransformation.initiate_data_transformation()
        logging.info("Data transformation completed")
        print(datatransformationartifact)

        logging.info("Model trainer component started")
        modeltrainerconfig = ModelTrainerConfig(training_pipeline_config=trainingpipelineconfig)
        modeltrainer = ModelTrainer(model_trainer_config=modeltrainerconfig, data_transformation_artifact=datatransformationartifact)
        modeltrainerartifact = modeltrainer.initiate_model_trainer()
        logging.info("Model trainer component completed")
        print(modeltrainerartifact)


    except Exception as e:
        raise NetworkSecurityException(e, sys)