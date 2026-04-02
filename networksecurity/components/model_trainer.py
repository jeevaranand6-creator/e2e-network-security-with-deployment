import os
import sys
from urllib.parse import urlparse

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact, ModelTrainerArtifact,DataTransformationArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import save_object,load_object,load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_metric
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import dagshub
# dagshub.init(repo_owner='jeevaranand6-creator', repo_name='e2e-network-security-with-deployment', mlflow=True)

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/jeevaranand6-creator/e2e-network-security-with-deployment.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="jeevaranand6-creator"
os.environ["MLFLOW_TRACKING_PASSWORD"]="86a778742516710f8becf874264177d7cc13933b"

import mlflow

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def track_mlflow(self, model_name: str, model, classification_metric_train_artifact: ClassificationMetricArtifact, classification_metric_test_artifact: ClassificationMetricArtifact):
        mlflow.set_registry_uri("https://dagshub.com/jeevaranand6-creator/e2e-network-security-with-deployment.mlflow")
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("train_accuracy", classification_metric_train_artifact.accuracy)
            mlflow.log_metric("train_precision", classification_metric_train_artifact.precision_score)
            mlflow.log_metric("train_recall", classification_metric_train_artifact.recall_score)
            mlflow.log_metric("train_f1_score", classification_metric_train_artifact.f1_score)
            mlflow.log_metric("test_accuracy", classification_metric_test_artifact.accuracy)
            mlflow.log_metric("test_precision", classification_metric_test_artifact.precision_score)
            mlflow.log_metric("test_recall", classification_metric_test_artifact.recall_score)
            mlflow.log_metric("test_f1_score", classification_metric_test_artifact.f1_score)
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name)
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
            mlflow.end_run()
        
    def train_model(self, x_train, y_train, x_test, y_test) -> NetworkModel:
        try:
            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Ada Boost": AdaBoostClassifier()
            }
            params = {
                "Logistic Regression": {},
                "Decision Tree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    # 'max_depth': [None, 10, 20, 30],
                    # 'min_samples_split': [2, 5, 10]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    # 'criterion': ['gini', 'entropy'],
                    # 'max_depth': [None, 10, 20],
                    # 'min_samples_split': [2, 5, 10]
                },
                "Gradient Boosting": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.001, 0.01, 0.1, 0.5],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
                    # 'max_depth': [3, 5]
                },
                "Ada Boost": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [0.001, 0.01, 0.1, 0.5]
                }
            }
            
            print(f"Training models: {list(models.keys())}")
            model_report: dict = evaluate_models(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, params=params)
            print(f"Model evaluation report: {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            classification_metric_test_artifact = get_classification_metric(y_true=y_test, y_pred=best_model.predict(x_test))
            classification_metric_train_artifact = get_classification_metric(y_true=y_train, y_pred=best_model.predict(x_train))

            ## Track the ML flow
            self.track_mlflow(model_name=best_model_name, model=best_model, classification_metric_train_artifact=classification_metric_train_artifact, classification_metric_test_artifact=classification_metric_test_artifact)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir, exist_ok=True)

            network_model = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

            save_object("final_models/final_model.pkl", best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_metric_train_artifact,
                test_metric_artifact=classification_metric_test_artifact
            )

            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(f"Loading transformed training and testing arrays.")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr = load_numpy_array_data(transformed_train_file_path)
            test_arr = load_numpy_array_data(transformed_test_file_path)

            x_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            return self.train_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


        except Exception as e:
            raise NetworkSecurityException(e, sys)
    

