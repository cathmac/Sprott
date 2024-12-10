import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

class AnomalyDetectionModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.feature_importance = None
        
    def prepare_features_target(self, df):
        """Prepare features and target from processed data."""
        # Get feature columns (excluding target and metadata)
        feature_cols = [col for col in df.columns if col.endswith('_encoded') 
                       or col in ['usd_amount', 'daily_amount_sum', 'approval_delay_hours',
                                'daily_transaction_count', 'hour', 'day_of_week',
                                'is_weekend', 'is_business_hours', 'month',
                                'self_approval_risk']]
        
        X = df[feature_cols]
        y = df['is_anomaly']
        
        return X, y
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model."""
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance."""
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5)
        
        return {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'cv_scores': cv_scores,
            'feature_importance': self.feature_importance
        }
    
    def plot_results(self, results):
        """Plot evaluation results."""
        output_dir = 'model_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Model Evaluation Results', fontsize=16)
        
        # Plot confusion matrix
        sns.heatmap(
            results['confusion_matrix'],
            annot=True,
            fmt='d',
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Confusion Matrix')
        
        # Plot ROC curve
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].plot(
            results['fpr'], 
            results['tpr'], 
            label=f'ROC (AUC = {results["roc_auc"]:.2f})'
        )
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        
        # Plot feature importance
        important_features = results['feature_importance'].head(10)
        sns.barplot(
            data=important_features,
            x='importance',
            y='feature',
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Top 10 Feature Importance')
        
        # Plot cross-validation scores
        sns.boxplot(data=results['cv_scores'], ax=axes[1, 1])
        axes[1, 1].set_title('Cross-validation Scores')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_evaluation.png')
        plt.close()
    
    def save_model(self, output_dir='model_output'):
        """Save the trained model and results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f'{output_dir}/anomaly_detection_model.joblib')
        
        # Save feature importance
        self.feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        
        print(f"\nModel saved to {output_dir}")

def main():
    # Load processed data
    print("Loading processed data...")
    df = pd.read_csv('processed_data/processed_transactions.csv')
    
    # Initialize model
    model = AnomalyDetectionModel()
    
    # Prepare features and target
    X, y = model.prepare_features_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model.train_model(X_train, y_train)
    
    # Evaluate model
    results = model.evaluate_model(X_test, y_test)
    
    # Plot and save results
    model.plot_results(results)
    
    # Save model
    model.save_model()

if __name__ == "__main__":
    main()