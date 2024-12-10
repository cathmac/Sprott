import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
import os
from doa_validator import DOAValidator

class TransactionPredictor:
    def __init__(self):
        # Load your pre-trained model
        model_dir = 'model_output'
        self.model = joblib.load(os.path.join(model_dir, 'anomaly_detection_model.joblib'))
        
        # Initialize DOA validator
        self.doa_validator = DOAValidator()
        
        # Example thresholds (adjust as necessary)
        self.risk_thresholds = {
            'low': 0.5,
            'medium': 0.7,
            'high': 0.85
        }

        # Define risk factors
        # Add or adjust factors to match what was used in training, if any
        self.risk_factors = {
            'large_amount': {
                'check': lambda tx: float(tx['usd_amount']) > 10000,
                'message': "Large transaction amount",
                'severity': 'high'
            }
            # Add more factors if they were part of training
        }

        self.transaction_history = []
        self.max_history = 1000

    def evaluate_risk_factors(self, transaction):
        risk_factors = []
        total_risk_score = 0
        for factor_name, factor in self.risk_factors.items():
            try:
                if factor['check'](transaction):
                    severity_scores = {'low': 0.2, 'medium': 0.5, 'high': 1.0}
                    risk_score = severity_scores[factor['severity']]
                    risk_factors.append({
                        'factor': factor_name,
                        'message': factor['message'],
                        'severity': factor['severity'],
                        'risk_score': risk_score
                    })
                    total_risk_score += risk_score
            except Exception as e:
                print(f"Error: {e}")  # Close the opening parenthesis
