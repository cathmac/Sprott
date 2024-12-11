import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


class AnomalyDetectionModel:
    def __init__(self, config=None, model_type="RandomForest"):
        self.config = config
        self.model_type = model_type
        self.model = None

    def train_model(self, X_train, y_train):
        print(f"Training {self.model_type} model with hyperparameter tuning...")

        # Handle missing values by imputing only numeric columns
        numeric_columns = X_train.select_dtypes(include=[np.number]).columns

        if X_train[numeric_columns].isnull().any().any():
            print("Missing values detected in numeric columns of X_train. Imputing with mean values...")
            X_train[numeric_columns] = X_train[numeric_columns].fillna(X_train[numeric_columns].mean())

        # Drop rows with any remaining NaNs in X_train and ensure y_train is updated accordingly
        initial_length = len(X_train)
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]

        print(f"Dropped {initial_length - len(X_train)} rows due to remaining missing values.")

        # Check the class distribution in y_train
        class_counts = y_train.value_counts()
        print(f"Class distribution in y_train after dropping missing values:\n{class_counts}")

        if len(class_counts) < 2:
            print("Warning: y_train contains only one class. Skipping SMOTE and training may not be meaningful.")
            return

        # Apply SMOTE for balancing classes
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except ValueError as e:
            print(f"SMOTE failed due to: {e}")
            return

        # Define hyperparameter grids
        if self.model_type == "RandomForest":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            model = RandomForestClassifier(random_state=42)

        elif self.model_type == "LogisticRegression":
            param_grid = {
                'C': [0.1, 1],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
            model = LogisticRegression(random_state=42, max_iter=1000)

        elif self.model_type == "XGBoost":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
            }
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Create a pipeline for scaling (if needed) and model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])

        # Perform grid search
        try:
            grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            print(f"Best parameters for {self.model_type}: {grid_search.best_params_}")
            print(f"{self.model_type} model training complete.")
        except Exception as e:
            print(f"Training failed due to: {e}")
            self.model = None

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            print("Error: Model has not been trained. Cannot evaluate.")
            return None

        y_pred = self.model.predict(X_test)

        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred, zero_division=0)

        return {
            'roc_auc': roc_auc,
            'confusion_matrix': conf_matrix,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report
        }

    def plot_results(self, results):
        cm = results['confusion_matrix']
        plt.matshow(cm, cmap='Blues')
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, f'{val}', ha='center', va='center')
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    def save_model(self, output_dir):
        if self.model is not None:
            model_path = os.path.join(output_dir, f"{self.model_type}_model.pkl")
            joblib.dump(self.model, model_path)
            print(f"Model saved to: {model_path}")
        else:
            print("No model to save.")

