import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random

class TransactionGenerator:
    def __init__(self, start_date, end_date, num_employees, seed=None):
        """
        Initializes the TransactionGenerator with date range and employee pool.
        
        Parameters:
        - start_date (str): Start date in 'YYYY-MM-DD' format.
        - end_date (str): End date in 'YYYY-MM-DD' format.
        - num_employees (int): Number of unique employees.
        - seed (int, optional): Random seed for reproducibility.
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.num_employees = num_employees
        self.employee_ids = [f"EMP{str(i).zfill(4)}" for i in range(1, num_employees + 1)]
        self.roles = ['ENG-JR', 'ENG-SR', 'TECH-LEAD', 'ENG-MGR', 'DIR', 'VP', 'EXEC', 'CEO', 'BOARD']
        self.departments = ['Software', 'Hardware', 'Cloud Services', 'Office Supplies', 'Training']
        self.merchants = ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple', 'JetBrains', 'Atlassian', 'UnknownMerchant1', 'UnknownMerchant2']
        self.merchant_regions = ['North America', 'Europe', 'Asia', 'South America', 'Africa', 'Australia']
        self.categories = ['Software', 'Hardware', 'Cloud Services', 'Office Supplies', 'Training', 'UnknownCategory1', 'UnknownCategory2']
        self.mcc_codes = [5411, 5812, 5732, 5944, 4789]  # Example MCC codes
        self.currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CAD']
        self.transaction_types = ['Purchase', 'Withdrawal', 'Refund', 'Transfer']
        self.transaction_countries = ['USA', 'Germany', 'UK', 'Japan', 'Canada']
        self.merchant_countries = ['USA', 'Germany', 'UK', 'Japan', 'Canada', 'UnknownCountry1', 'UnknownCountry2']
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def generate_dataset(self, num_transactions, anomaly_ratio=0.05):
        """
        Generates a synthetic dataset of credit card transactions.
        
        Parameters:
        - num_transactions (int): Number of transactions to generate.
        - anomaly_ratio (float): Proportion of transactions that are anomalous.
        
        Returns:
        - pd.DataFrame: Generated dataset.
        """
        self.logger.info("Generating synthetic transaction dataset...")
        try:
            # Initialize lists to store transaction data
            transaction_ids = [f"TXN{str(i).zfill(6)}" for i in range(1, num_transactions + 1)]
            timestamps = [self._random_timestamp() for _ in range(num_transactions)]
            employee_ids = [random.choice(self.employee_ids) for _ in range(num_transactions)]
            transaction_roles = [self._get_role(emp_id) for emp_id in employee_ids]
            transaction_departments = [self._get_department(role) for role in transaction_roles]
            merchants = [random.choice(self.merchants) for _ in range(num_transactions)]
            merchant_regions = [random.choice(self.merchant_regions) for _ in range(num_transactions)]
            categories = [random.choice(self.categories) for _ in range(num_transactions)]
            mcc = [random.choice(self.mcc_codes) for _ in range(num_transactions)]
            amounts = [round(random.uniform(10, 50000), 2) for _ in range(num_transactions)]
            currencies = [random.choice(self.currencies) for _ in range(num_transactions)]
            usd_amounts = [self._convert_to_usd(amount, currency) for amount, currency in zip(amounts, currencies)]
            transaction_types = [random.choice(self.transaction_types) for _ in range(num_transactions)]
            needs_approval = [random.choice([True, False]) for _ in range(num_transactions)]
            approver_ids = [self._get_approver(emp_id) for emp_id in employee_ids]
            self_approved = [random.choice([True, False]) for _ in range(num_transactions)]
            is_anomaly = self._assign_anomalies(num_transactions, anomaly_ratio)
            anomaly_types = [self._get_anomaly_type() if anomaly else np.nan for anomaly in is_anomaly]
            approval_timestamps = [self._get_approval_timestamp(ts) if needs else pd.NaT for ts, needs in zip(timestamps, needs_approval)]
            approver_roles = [self._get_role(approver_id) if approver_id != 'NONE' else 'NONE' for approver_id in approver_ids]
            board_approval = [random.choice([True, False]) if role != 'BOARD' else False for role in transaction_roles]
            transaction_country = [random.choice(self.transaction_countries) for _ in range(num_transactions)]
            merchant_country = [random.choice(self.merchant_countries) for _ in range(num_transactions)]
            
            # Create DataFrame
            data = {
                'transaction_id': transaction_ids,
                'timestamp': timestamps,
                'employee_id': employee_ids,  # Renamed for consistency
                'employee_role': transaction_roles,
                'department': transaction_departments,
                'merchant': merchants,
                'merchant_region': merchant_regions,
                'category': categories,
                'mcc': mcc,
                'amount': amounts,
                'usd_amount': usd_amounts,
                'currency': currencies,
                'transaction_type': transaction_types,
                'needs_approval': needs_approval,
                'approver_id': approver_ids,
                'self_approved': self_approved,
                'is_anomaly': is_anomaly,
                'anomaly_type': anomaly_types,
                'approval_timestamp': approval_timestamps,
                'approver_role': approver_roles,
                'board_approval': board_approval,
                'transaction_country': transaction_country,
                'merchant_country': merchant_country
            }
            
            # Debugging: Ensure all columns have the correct length
            self.logger.debug("Lengths of all columns before DataFrame creation:")
            for col, lst in data.items():
                self.logger.debug(f"{col}: {len(lst)}")
            
            # Verify that all lists have the same length
            lengths = [len(lst) for lst in data.values()]
            if len(set(lengths)) != 1:
                mismatched_lengths = {}
                for col, lst in data.items():
                    mismatched_lengths[col] = len(lst)
                self.logger.error(f"Mismatch in column lengths: {mismatched_lengths}")
                raise ValueError("All arrays must be of the same length")
            
            df = pd.DataFrame(data)
            self.logger.info("Dataset generation completed.")
            return df
        except Exception as e:
            self.logger.error(f"Error during dataset generation: {str(e)}")
            return None
    
    def _random_timestamp(self):
        """Generates a random timestamp between start_date and end_date."""
        delta = self.end_date - self.start_date
        int_delta = delta.days * 24 * 60 * 60 + delta.seconds
        random_second = random.randrange(int_delta)
        return self.start_date + timedelta(seconds=random_second)
    
    def _get_role(self, emp_id):
        """Assigns a role based on employee ID."""
        # Simple example: Assign roles randomly
        return random.choice(self.roles)
    
    def _get_department(self, role):
        """Assigns a department based on role."""
        # Simple example: Map roles to departments
        role_department_map = {
            'ENG-JR': 'Software',
            'ENG-SR': 'Software',
            'TECH-LEAD': 'Cloud Services',
            'ENG-MGR': 'Office Supplies',
            'DIR': 'Hardware',
            'VP': 'Hardware',
            'EXEC': 'Training',
            'CEO': 'Executive',
            'BOARD': 'Board'
        }
        return role_department_map.get(role, 'Unknown')
    
    def _get_approver(self, emp_id):
        """Assigns an approver ID based on employee ID."""
        # Simple example: Assign 'NONE' or another employee
        if random.random() < 0.1:  # 10% chance of no approver
            return 'NONE'
        else:
            return random.choice(self.employee_ids)
    
    def _convert_to_usd(self, amount, currency):
        """Converts amount to USD based on currency."""
        conversion_rates = {
            'USD': 1.0,
            'EUR': 1.1,
            'GBP': 1.3,
            'JPY': 0.009,
            'CAD': 0.75
        }
        return round(amount * conversion_rates.get(currency, 1.0), 2)
    
    def _assign_anomalies(self, num_transactions, anomaly_ratio):
        """Randomly assigns anomalies based on the specified ratio."""
        num_anomalies = int(num_transactions * anomaly_ratio)
        anomalies = [False] * num_transactions
        if num_anomalies > 0:
            anomaly_indices = random.sample(range(num_transactions), num_anomalies)
            for idx in anomaly_indices:
                anomalies[idx] = True
        return anomalies
    
    def _get_anomaly_type(self):
        """Randomly selects an anomaly type."""
        anomaly_types = ['Amount Spike', 'Time Anomaly', 'Merchant Anomaly', 'Category Anomaly']
        return random.choice(anomaly_types)
    
    def _get_approval_timestamp(self, transaction_timestamp):
        """Generates an approval timestamp based on transaction timestamp."""
        if pd.isna(transaction_timestamp):
            return pd.NaT
        # Approval is within 2 hours after transaction
        approval_delay = timedelta(minutes=random.randint(1, 120))
        return transaction_timestamp + approval_delay
