import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
import pandas as pd
import joblib
import os
import numpy as np


class AnomalyDetectionModel:
    def __init__(self, config=None, model_type="IsolationForest"):
        self.config = config
        self.model_type = model_type
        self.model = None

    def train_model(self, X_train, y_train=None):
        print(f"Training {self.model_type} model...")
        if self.model_type == "IsolationForest":
            self.model = IsolationForest(contamination=self.config.get('anomaly_ratio', 0.02), n_estimators=300, random_state=42)
        elif self.model_type == "OneClassSVM":
            self.model = OneClassSVM(nu=self.config.get('anomaly_ratio', 0.02), kernel='rbf', gamma='auto')
        elif self.model_type == "LOF":
            self.model = LocalOutlierFactor(novelty=True, n_neighbors=20, contamination=self.config.get('anomaly_ratio', 0.02))
        elif self.model_type == "DBSCAN":
            self.model = DBSCAN(eps=0.3, min_samples=10)
        elif self.model_type == "EllipticEnvelope":
            self.model = EllipticEnvelope(contamination=self.config.get('anomaly_ratio', 0.02))
        elif self.model_type == "KMeans":
            self.model = KMeans(n_clusters=2, random_state=42)
        elif self.model_type == "GaussianMixture":
            self.model = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        elif self.model_type == "LogisticRegression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == "RandomForest":
            self.model = RandomForestClassifier(n_estimators=300, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if self.model_type in ["RandomForest", "LogisticRegression"]:
            self.model.fit(X_train, y_train)
        elif self.model_type == "LOF" and self.model.novelty:
            self.model.fit(X_train)
        else:
            self.model.fit(X_train)

        print(f"{self.model_type} model training complete.")

    def evaluate_model(self, X_test, y_test):
        # Handle models without a predict method
        if self.model_type == "DBSCAN":
            y_pred = self.model.fit_predict(X_test)
        elif self.model_type == "LOF" and self.model.novelty:
            y_pred = self.model.predict(X_test)
        else:
            y_pred = self.model.predict(X_test)
            y_pred = [0 if pred == -1 else 1 for pred in y_pred]

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y
