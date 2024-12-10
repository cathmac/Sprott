import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

class ThresholdOptimizer:
    def __init__(self):
        self.threshold_metrics = {}
        self.optimal_thresholds = {}
        
    def optimize_thresholds(self, y_true, y_prob, cost_matrix=None):
        """
        Optimize thresholds based on multiple criteria.
        
        Parameters:
        - y_true: True labels
        - y_prob: Predicted probabilities
        - cost_matrix: Dictionary with costs for FP and FN
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
        return {
            'threshold': thresholds[optimal_idx],
            'precision': precisions[optimal_idx],
            'recall': recalls[optimal_idx]
        }
    
    def _find_cost_optimal_threshold(self, y_true, y_prob, cost_matrix):
        """Find threshold that minimizes total cost."""
        thresholds = np.linspace(0, 1, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            total_cost = fp * cost_matrix['fp_cost'] + fn * cost_matrix['fn_cost']
            costs.append(total_cost)
        
        optimal_idx = np.argmin(costs)
        return {
            'threshold': thresholds[optimal_idx],
            'total_cost': costs[optimal_idx]
        }
    
    def _find_f1_optimal_threshold(self, y_true, y_prob):
        """Find threshold that maximizes F1 score."""
        thresholds = np.linspace(0, 1, 100)
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
            return {'threshold': 0.99, 'actual_precision': precisions[-1]}
        
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
            return {'threshold': 0.01, 'actual_recall': recalls[0]}
        
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
        plt.plot(recalls, precisions)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        
        # Plot 2: ROC curve
        plt.subplot(2, 2, 2)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        
        # Plot 3: Threshold distribution
        plt.subplot(2, 2, 3)
        plt.hist(y_prob, bins=50)
        for name, metrics in self.optimal_thresholds.items():
            plt.axvline(metrics['threshold'], label=name)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Score Distribution and Thresholds')
        plt.legend()
        
        plt.tight_layout()
        return plt.gcf()