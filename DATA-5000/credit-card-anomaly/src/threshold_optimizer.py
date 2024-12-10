import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import joblib
import warnings

class ThresholdOptimizer:
    def __init__(self, config=None):
        """
        Initializes the ThresholdOptimizer.
        
        Parameters:
        - config (dict, optional): Configuration for severity weights and thresholds.
        """
        self.threshold_metrics = {}
        self.optimal_thresholds = {}
        self.severity_weights = config.get('severity_weights') if config and 'severity_weights' in config else {'low': 0.2, 'medium': 0.5, 'high': 1.0}
    
    def optimize_thresholds(self, y_true, y_prob, cost_matrix=None):
        """
        Optimize thresholds based on multiple criteria.
        
        Parameters:
        - y_true (array-like): True binary labels.
        - y_prob (array-like): Predicted probabilities for the positive class.
        - cost_matrix (dict, optional): Dictionary with costs for FP and FN.
                                         Example: {'fp_cost': 1.0, 'fn_cost': 5.0}
        
        Returns:
        - dict: Optimal thresholds for each criterion.
        """
        if cost_matrix is None:
            cost_matrix = {
                'fp_cost': 1.0,  # Cost of false positive
                'fn_cost': 5.0   # Cost of false negative (usually higher)
            }
        
        # Calculate various optimal thresholds
        self.optimal_thresholds = {
            'balanced': self._find_balanced_threshold(y_true, y_prob),
            'cost_optimal': self._find_cost_optimal_threshold(y_true, y_prob, cost_matrix),
            'f1_optimal': self._find_f1_optimal_threshold(y_true, y_prob),
            'precision_focused': self._find_precision_threshold(y_true, y_prob, target_precision=0.95),
            'recall_focused': self._find_recall_threshold(y_true, y_prob, target_recall=0.90)
        }
        
        return self.optimal_thresholds
    
    def _find_balanced_threshold(self, y_true, y_prob):
        """Find threshold that balances precision and recall."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        scores = np.abs(precisions - recalls)
        optimal_idx = np.argmin(scores)
        if optimal_idx >= len(thresholds):
            optimal_idx = len(thresholds) - 1
        return {
            'threshold': thresholds[optimal_idx],
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx]
        }
    
    def _find_cost_optimal_threshold(self, y_true, y_prob, cost_matrix):
        """Find threshold that minimizes total cost using vectorization."""
        thresholds = np.linspace(0, 1, 1000)
        y_pred = y_prob[:, None] >= thresholds  # Shape: (n_samples, n_thresholds)
        fp = np.sum((y_pred & (y_true[:, None] == 0)), axis=0)
        fn = np.sum((~y_pred & (y_true[:, None] == 1)), axis=0)
        total_cost = fp * cost_matrix['fp_cost'] + fn * cost_matrix['fn_cost']
        optimal_idx = np.argmin(total_cost)
        return {
            'threshold': thresholds[optimal_idx],
            'total_cost': total_cost[optimal_idx]
        }
    
    def _find_f1_optimal_threshold(self, y_true, y_prob):
        """Find threshold that maximizes F1 score."""
        thresholds = np.linspace(0, 1, 1000)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            f1_scores.append(f1)
        
        optimal_idx = np.argmax(f1_scores)
        return {
            'threshold': thresholds[optimal_idx],
            'f1_score': f1_scores[optimal_idx]
        }
    
    def _find_precision_threshold(self, y_true, y_prob, target_precision=0.95):
        """Find threshold that achieves target precision."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        valid_idx = precisions >= target_precision
        if not any(valid_idx):
            warnings.warn("Desired precision not achievable. Using highest available precision threshold.")
            return {'threshold': thresholds[-1] if len(thresholds) > 0 else 0.99, 'actual_precision': precisions[-1] if len(precisions) > 0 else 0.0}
        
        optimal_idx = np.where(valid_idx)[0][0]
        return {
            'threshold': thresholds[optimal_idx],
            'actual_precision': precisions[optimal_idx]
        }
    
    def _find_recall_threshold(self, y_true, y_prob, target_recall=0.90):
        """Find threshold that achieves target recall."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        valid_idx = recalls >= target_recall
        if not any(valid_idx):
            warnings.warn("Desired recall not achievable. Using lowest available recall threshold.")
            return {'threshold': thresholds[0] if len(thresholds) > 0 else 0.01, 'actual_recall': recalls[0] if len(recalls) > 0 else 0.0}
        
        optimal_idx = np.where(valid_idx)[0][-1]
        return {
            'threshold': thresholds[optimal_idx],
            'actual_recall': recalls[optimal_idx]
        }
    
    def plot_threshold_analysis(self, y_true, y_prob):
        """Plot threshold analysis curves."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Precision-Recall curve
        plt.subplot(2, 2, 1)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        plt.plot(recalls, precisions, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        
        # Plot 2: ROC curve
        plt.subplot(2, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # Plot 3: Threshold distribution
        plt.subplot(2, 2, 3)
        plt.hist(y_prob, bins=50, alpha=0.7, label='Predicted Probabilities')
        for name, metrics in self.optimal_thresholds.items():
            plt.axvline(metrics['threshold'], label=name)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Score Distribution and Thresholds')
        plt.legend()
        
        # Plot 4: F1 Score vs Threshold
        plt.subplot(2, 2, 4)
        f1_scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in np.linspace(0, 1, 1000)]
        thresholds = np.linspace(0, 1, 1000)
        plt.plot(thresholds, f1_scores, label='F1 Score')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()
    
    def save_thresholds(self, file_path):
        """
        Save the optimal thresholds to a file.
        
        Parameters:
        - file_path (str): Path to save the thresholds.
        """
        joblib.dump(self.optimal_thresholds, file_path)
        print(f"Optimal thresholds saved to {file_path}")
    
    def load_thresholds(self, file_path):
        """
        Load the optimal thresholds from a file.
        
        Parameters:
        - file_path (str): Path to load the thresholds from.
        
        Returns:
        - dict: Loaded optimal thresholds.
        """
        self.optimal_thresholds = joblib.load(file_path)
        print(f"Optimal thresholds loaded from {file_path}")
        return self.optimal_thresholds
