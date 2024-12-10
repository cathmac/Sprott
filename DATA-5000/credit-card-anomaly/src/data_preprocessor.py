import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from datetime import datetime

class CreditCardDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, file_path):
        """Load the credit card transaction data."""
        print("Loading data from:", /Users/cathmac/Github/Sprott/DATA-5000/credit-card-anomaly/src/transaction_analysis_output/credit_card_transactions.csv)
        return pd.read_csv(/Users/cathmac/Github/Sprott/DATA-5000/credit-card-anomaly/src/transaction_analysis_output/credit_card_transactions.csv)
        
    def clean_data(self, df):
        """Clean the raw transaction data."""
        print("Cleaning data...")
        df_cleaned = df.copy()
        
        # Convert timestamps to datetime
        df_cleaned['timestamp'] = pd.to_datetime(df_cleaned['timestamp'])
        df_cleaned['approval_timestamp'] = pd.to_datetime(df_cleaned['approval_timestamp'])
        
        # Handle missing values
        df_cleaned['approver_id'].fillna('NONE', inplace=True)
        df_cleaned['approval_timestamp'].fillna(pd.NaT, inplace=True)
        
        # Convert boolean fields
        boolean_fields = ['needs_approval', 'self_approved', 'is_anomaly']
        for field in boolean_fields:
            df_cleaned[field] = df_cleaned[field].astype(bool)
        
        # Ensure numeric fields are correct type
        df_cleaned['usd_amount'] = pd.to_numeric(df_cleaned['usd_amount'], errors='coerce')
        
        return df_cleaned
        
    def engineer_features(self, df):
        """Create new features from the transaction data."""
        print("Engineering features...")
        df_featured = df.copy()
        
        # Time-based features
        df_featured['hour'] = df_featured['timestamp'].dt.hour
        df_featured['day_of_week'] = df_featured['timestamp'].dt.dayofweek
        df_featured['is_weekend'] = df_featured['timestamp'].dt.dayofweek >= 5
        df_featured['is_business_hours'] = (df_featured['hour'] >= 9) & (df_featured['hour'] <= 17)
        df_featured['month'] = df_featured['timestamp'].dt.month
        
        # Transaction patterns
        df_featured['daily_transaction_count'] = df_featured.groupby(
            [df_featured['timestamp'].dt.date, 'employee_id']
        )['transaction_id'].transform('count')
        
        # Amount patterns
        df_featured['daily_amount_sum'] = df_featured.groupby(
            [df_featured['timestamp'].dt.date, 'employee_id']
        )['usd_amount'].transform('sum')
        
        # Approval patterns
        mask = df_featured['approval_timestamp'].notna()
        df_featured['approval_delay_hours'] = pd.NA
        df_featured.loc[mask, 'approval_delay_hours'] = (
            df_featured.loc[mask, 'approval_timestamp'] - 
            df_featured.loc[mask, 'timestamp']
        ).dt.total_seconds() / 3600
        
        # Self-approval risk
        df_featured['self_approval_risk'] = (
            df_featured['self_approved'] & 
            df_featured['needs_approval']
        ).astype(int)
        
        return df_featured
    
    def encode_categorical_features(self, df):
        """Encode categorical variables for model input."""
        print("Encoding categorical features...")
        df_encoded = df.copy()
        
        categorical_features = [
            'employee_role', 
            'department', 
            'merchant_region', 
            'category', 
            'currency', 
            'transaction_type'
        ]
        
        for feature in categorical_features:
            le = LabelEncoder()
            df_encoded[f'{feature}_encoded'] = le.fit_transform(df_encoded[feature].fillna('MISSING'))
            self.label_encoders[feature] = le
        
        return df_encoded
    
    def scale_numerical_features(self, df):
        """Scale numerical features."""
        print("Scaling numerical features...")
        numerical_features = [
            'usd_amount',
            'daily_amount_sum',
            'approval_delay_hours',
            'daily_transaction_count'
        ]
        
        # Create copy of dataframe
        df_scaled = df.copy()
        
        # Fill missing values with -1 before scaling
        for feature in numerical_features:
            df_scaled[feature] = df_scaled[feature].fillna(-1)
        
        # Scale features
        df_scaled[numerical_features] = self.scaler.fit_transform(df_scaled[numerical_features])
        
        return df_scaled
    
    def get_feature_columns(self):
        """Get list of feature columns for model training."""
        base_features = [
            'usd_amount',
            'daily_amount_sum',
            'approval_delay_hours',
            'daily_transaction_count',
            'hour',
            'day_of_week',
            'is_weekend',
            'is_business_hours',
            'month',
            'self_approval_risk'
        ]
        
        encoded_features = [
            'employee_role_encoded',
            'department_encoded',
            'merchant_region_encoded',
            'category_encoded',
            'currency_encoded',
            'transaction_type_encoded'
        ]
        
        return base_features + encoded_features
    
    def process_data(self, input_file, output_dir='processed_data'):
        """Complete data processing pipeline."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        df = self.load_data(input_file)
        
        # Process data
        df_cleaned = self.clean_data(df)
        df_featured = self.engineer_features(df_cleaned)
        df_encoded = self.encode_categorical_features(df_featured)
        df_scaled = self.scale_numerical_features(df_encoded)
        
        # Save processed data
        output_file = os.path.join(output_dir, 'processed_transactions.csv')
        df_scaled.to_csv(output_file, index=False)
        print(f"Processed data saved to: {output_file}")
        
        # Generate processing summary
        summary = {
            'total_transactions': len(df),
            'total_employees': df['employee_id'].nunique(),
            'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}",
            'anomaly_rate': df['is_anomaly'].mean(),
            'features_generated': len(self.get_feature_columns()),
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save summary
        summary_file = os.path.join(output_dir, 'processing_summary.txt')
        with open(summary_file, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        return df_scaled

if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = CreditCardDataPreprocessor()
    
    # Process data
    processed_df = preprocessor.process_data(
        input_file='transaction_analysis_output/credit_card_transactions.csv'
    )
    
    print("\nProcessing complete!")
    print(f"Generated features: {preprocessor.get_feature_columns()}")