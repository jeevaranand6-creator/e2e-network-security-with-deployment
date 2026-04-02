from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from sklearn.metrics import f1_score, precision_score, recall_score, r2_score, accuracy_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys

def get_classification_metric(y_true, y_pred) -> ClassificationMetricArtifact:
        try:
            f1_score_value = f1_score(y_true, y_pred)
            precision_score_value = precision_score(y_true, y_pred)
            recall_score_value = recall_score(y_true, y_pred)
            r2_score_value = r2_score(y_true, y_pred)
            accuracy_value = accuracy_score(y_true, y_pred)
            classification_metric_artifact = ClassificationMetricArtifact(
                f1_score=f1_score_value,
                precision_score=precision_score_value,
                recall_score=recall_score_value,
                r2_score=r2_score_value,
                accuracy=accuracy_value
            )

            return classification_metric_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)