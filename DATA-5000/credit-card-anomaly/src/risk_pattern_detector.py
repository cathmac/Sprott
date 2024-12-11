import pandas as pd
import logging

class RiskPatternDetector:
    def __init__(self, config=None):
        """
        Initializes the RiskPatternDetector with configuration settings.
        
        Parameters:
        - config (dict, optional): Configuration dictionary containing settings for risk detection.
        """
        self.config = config if config is not None else {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def detect_risks(self, df):
        """
        Detects high-risk transactions based on predefined patterns.
        
        Parameters:
        - df (pd.DataFrame): DataFrame containing transaction data.
        
        Returns:
        - pd.DataFrame: DataFrame with additional columns indicating risk flags.
        """
        try:
            self.logger.info("Detecting high-risk transactions based on patterns...")
            df = df.copy()
            
            # Ensure 'timestamp' is timezone-aware. If not, localize it to UTC.
            if df['timestamp'].dt.tz is None:
                self.logger.info("Localizing 'timestamp' to UTC as it is timezone-naive.")
                df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            else:
                self.logger.info("'timestamp' is already timezone-aware.")
            
            # Convert to local timezone (US/Eastern)
            df['timestamp_local'] = df['timestamp'].dt.tz_convert('US/Eastern')
            df['hour'] = df['timestamp_local'].dt.hour
            business_start = self.config.get('business_hours_start', 6)  # Default 6 AM
            business_end = self.config.get('business_hours_end', 20)    # Default 8 PM
            unusual_hours = df['hour'].apply(lambda x: x < business_start or x > business_end)
            df['unusual_hours_risk'] = unusual_hours
            
            # High amount transactions
            amount_threshold = self.config.get('high_amount_threshold', 10000)  # Default threshold
            df['high_amount_risk'] = df['usd_amount'] > amount_threshold
            
            # Unusual merchant or category
            unusual_merchants = self.config.get('unusual_merchants', [])
            unusual_categories = self.config.get('unusual_categories', [])
            if unusual_merchants:
                df['unusual_merchant_risk'] = ~df['merchant'].isin(unusual_merchants)
            else:
                df['unusual_merchant_risk'] = False  # No unusual merchants defined
            
            if unusual_categories:
                df['unusual_category_risk'] = ~df['category'].isin(unusual_categories)
            else:
                df['unusual_category_risk'] = False  # No unusual categories defined
            
            # Combine all risk flags
            df['is_high_risk'] = (
                df['high_amount_risk'] |
                df['unusual_hours_risk'] |
                df['unusual_merchant_risk'] |
                df['unusual_category_risk']
            )
            
            # Drop intermediate columns
            df.drop(['timestamp_local', 'hour', 'high_amount_risk', 'unusual_hours_risk', 
                     'unusual_merchant_risk', 'unusual_category_risk'], axis=1, inplace=True)
            
            self.logger.info("Risk detection completed.")
            return df
        except Exception as e:
            self.logger.error(f"Error during risk detection: {str(e)}")
            raise e

