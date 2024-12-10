import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class AnomalyDetectionModel:
    def __init__(self, config=None):
        """
        Initializes the AnomalyDetectionModel with configuration settings.
        
        Parameters:
        - config (dict, optional): Configuration dictionary containing settings for the model.
        """
        self.config = config if config is not None else {}
        self.model = RandomForestClassifier(random_state=42)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def prepare_features_target(self, df):
        """
        Prepares the feature matrix X and target vector y for model training.
        
        Parameters:
        - df (pd.DataFrame): Processed transaction DataFrame.
        
        Returns:
        - X (pd.DataFrame): Feature matrix.
        - y (pd.Series): Target vector.
        """
        self.logger.info("Preparing features and target for the model...")
        # Example feature engineering
        X = df.copy()
        y = X['is_anomaly'].astype(int)
        
        # Drop columns that won't be used for modeling, including 'transaction_employee_ids'
        drop_cols = [
            'transaction_id', 'timestamp', 'transaction_employee_ids', 'approver_id', 'approval_timestamp',
            'is_anomaly', 'anomaly_type', 'violations', 'is_valid', 'self_approved',
            'board_approval', 'needs_approval'
        ]
        X.drop(columns=[col for col in drop_cols if col in X.columns], inplace=True)
        
        # Handle categorical variables
        categorical_cols = [
            'employee_role', 'department', 'merchant', 'merchant_region',
            'category', 'currency', 'transaction_type', 'transaction_country',
            'merchant_country'
        ]
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Handle missing values if any
        X.fillna(0, inplace=True)
        
        # Verify that no object/string columns remain
        string_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if string_columns:
            self.logger.warning(f"The following string columns were not encoded and will be dropped: {string_columns}")
            X.drop(columns=string_columns, inplace=True)
        
        self.logger.info("Feature preparation completed.")
        return X, y
    
    def train_model(self, X_train, y_train):
        """
        Trains the anomaly detection model.
        
        Parameters:
        - X_train (pd.DataFrame): Training feature matrix.
        - y_train (pd.Series): Training target vector.
        """
        self.logger.info("Training the anomaly detection model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed.")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluates the trained model on the test set.
        
        Parameters:
        - X_test (pd.DataFrame): Testing feature matrix.
        - y_test (pd.Series): Testing target vector.
        
        Returns:
        - dict: Evaluation metrics.
        """
        self.logger.info("Evaluating the model...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        self.logger.info(f"ROC AUC Score: {roc_auc}")
        self.logger.info(f"Classification Report:\n{report}")
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        return {
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_results(self, results):
        """
        Plots ROC Curve and Confusion Matrix.
        
        Parameters:
        - results (dict): Evaluation metrics.
        """
        self.logger.info("Plotting evaluation results...")
        
        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {results["roc_auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        roc_plot_file = os.path.join('transaction_analysis_output', 'roc_curve.png')
        plt.savefig(roc_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"ROC Curve saved to: {roc_plot_file}")
        
        # Confusion Matrix
        plt.figure(figsize=(6, 5))
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomalous'], yticklabels=['Normal', 'Anomalous'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        cm_plot_file = os.path.join('transaction_analysis_output', 'confusion_matrix.png')
        plt.savefig(cm_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion Matrix saved to: {cm_plot_file}")
    
    def save_model(self, output_dir):
        """
        Saves the trained model to disk.
        
        Parameters:
        - output_dir (str): Directory to save the model.
        """
        model_file = os.path.join(output_dir, 'anomaly_detection_model.joblib')
        joblib.dump(self.model, model_file)
        self.logger.info(f"Trained model saved to: {model_file}")
