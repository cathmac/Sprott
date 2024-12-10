# data_preprocessing.py

import pandas as pd
import numpy as np
import logging
from doa_validator import DOAValidator
from risk_pattern_detector import RiskPatternDetector
import os

class CreditCardDataPreprocessor:
    def __init__(self, config):
        """
        Initializes the CreditCardDataPreprocessor with necessary components.
        
        Parameters:
        - config (dict): Configuration dictionary containing settings for validators and detectors.
        """
        self.config = config
        self.validator = DOAValidator(config=config.get('doa_validator'))
        self.risk_detector = RiskPatternDetector(config=config.get('risk_pattern_detector'))
        
        # Initialize transaction history as an empty DataFrame
        self.transaction_history = pd.DataFrame()
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def process_data(self, input_file, output_dir):
        """
        Processes the raw transaction data by cleaning, validating, and detecting risks.
        
        Parameters:
        - input_file (str): Path to the raw transaction CSV file.
        - output_dir (str): Directory to save processed data and reports.
        
        Returns:
        - pd.DataFrame: Processed transaction DataFrame.
        """
        try:
            self.logger.info(f"Loading data from: {input_file}")
            df = pd.read_csv(input_file, parse_dates=['timestamp', 'approval_timestamp'])
            
            # Clean data
            self.logger.info("Cleaning data...")
            df_cleaned = df.copy()
            df_cleaned['approver_id'] = df_cleaned['approver_id'].fillna('NONE')
            df_cleaned['approval_timestamp'] = df_cleaned['approval_timestamp'].fillna(pd.NaT)
            
            # Rename 'transaction_employee_ids' to 'employee_id' for consistency
            if 'transaction_employee_ids' in df_cleaned.columns:
                df_cleaned.rename(columns={'transaction_employee_ids': 'employee_id'}, inplace=True)
                self.logger.info("Renamed 'transaction_employee_ids' to 'employee_id' for consistency.")
            
            # Validate transactions
            self.logger.info("Validating transactions against DOA policies...")
            validation_results = df_cleaned.apply(
                lambda row: pd.Series(
                    self.validator.validate_transaction(
                        row.to_dict(), 
                        transaction_history=self.transaction_history
                    ), 
                    index=['is_valid', 'violations']
                ),
                axis=1
            )
            df_cleaned = pd.concat([df_cleaned, validation_results], axis=1)
            
            # Update transaction history
            self.transaction_history = pd.concat([self.transaction_history, df_cleaned], ignore_index=True)
            
            # Detect risk patterns
            self.logger.info("Detecting risk patterns...")
            df_cleaned = self.risk_detector.detect_risks(df_cleaned)
            
            # Save processed data
            processed_data_file = os.path.join(output_dir, 'processed_transactions.csv')
            df_cleaned.to_csv(processed_data_file, index=False)
            self.logger.info(f"Processed transaction data saved to: {processed_data_file}")
            
            return df_cleaned
        
        except Exception as e:
            self.logger.error(f"Error during data preprocessing: {str(e)}")
            return None
