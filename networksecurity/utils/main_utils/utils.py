import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric
import numpy as np
import dill
import pickle
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_numpy_array_data(file_path: str, array: np.array) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info(f"Saving object file: {file_path}")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Exiting save_object method of MainUtils class")

    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise NetworkSecurityException(f"The file: {file_path} is not exists", sys)
        logging.info(f"Loading object file: {file_path}")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def evaluate_models(x_train, y_train, x_test, y_test, models: dict, params: dict) -> dict:
    try:
        model_report: dict = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = params.get(model_name, {})

            grid_search = GridSearchCV(estimator=model, param_grid=param, cv=3)
            logging.info(f"Training {model_name} model")
            grid_search.fit(x_train, y_train)

            model.set_params(**grid_search.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_metric = get_classification_metric(y_true=y_train, y_pred=y_train_pred)
            test_metric = get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            model_report[model_name] = test_metric.accuracy

        return model_report

    except Exception as e:
        raise NetworkSecurityException(e, sys)