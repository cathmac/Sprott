import unittest
from datetime import datetime
import pandas as pd
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from transaction_generator import TransactionGenerator

class TestTransactionGenerator(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        self.generator = TransactionGenerator(
            start_date='2024-01-03',
            end_date='2024-12-06',
            num_employees=100  # Smaller number for testing
        )

    def test_employee_generation(self):
        """Test that employee profiles are generated correctly."""
        self.assertEqual(len(self.generator.employees), 100)
        
        # Test employee structure
        employee = self.generator.employees[0]
        self.assertTrue('employee_id' in employee)
        self.assertTrue('role' in employee)
        self.assertTrue('department' in employee)
        self.assertTrue('delegation_limit' in employee)
        
        # Test employee ID format
        self.assertTrue(employee['employee_id'].startswith('EMP'))
        self.assertEqual(len(employee['employee_id']), 9)  # EMP + 6 digits

    def test_transaction_generation(self):
        """Test that transactions are generated correctly."""
        df = self.generator.generate_dataset(num_transactions=1000, anomaly_ratio=0.05)
        
        # Test basic properties
        self.assertEqual(len(df), 1000)
        
        # Test required columns
        required_columns = [
            'transaction_id', 'timestamp', 'employee_id', 'amount', 
            'currency', 'usd_amount', 'merchant', 'merchant_region'
        ]
        for column in required_columns:
            self.assertTrue(column in df.columns)
        
        # Test date range
        self.assertTrue(all(
            datetime.strptime('2024-01-03', '%Y-%m-%d') <= pd.to_datetime(df['timestamp'])
        ))
        self.assertTrue(all(
            pd.to_datetime(df['timestamp']) <= datetime.strptime('2024-12-06', '%Y-%m-%d')
        ))

    def test_anomaly_ratio(self):
        """Test that the anomaly ratio is approximately correct."""
        df = self.generator.generate_dataset(num_transactions=1000, anomaly_ratio=0.05)
        anomaly_count = len(df[df['is_anomaly']])
        
        # Allow for some random variation (±1%)
        self.assertTrue(45 <= anomaly_count <= 55)