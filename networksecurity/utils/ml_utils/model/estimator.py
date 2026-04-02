from networksecurity.constants.training_pipeline import SAVED_MODEL_DIR, MODEL_TRAINER_TRAINED_MODEL_NAME
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys

from networksecurity.utils.main_utils.utils import save_object

class NetworkModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X):
        try:
            X = self.preprocessor.transform(X)
            return self.model.predict(X)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def save_model(self, model_dir: str, model_file_name: str) -> None:
        try:
            os.makedirs(model_dir, exist_ok=True)
            model_file_path = os.path.join(model_dir, model_file_name)
            save_object(file_path=model_file_path, obj=self.model)

        except Exception as e:
            raise NetworkSecurityException(e, sys)