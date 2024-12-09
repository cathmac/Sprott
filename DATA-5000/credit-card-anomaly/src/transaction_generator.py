import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class TransactionGenerator:
    def __init__(self, start_date='2024-01-03', end_date='2024-12-06', num_employees=3000):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.num_employees = num_employees
        
        # Define currencies and exchange rates
        self.currencies = {
            'USD': 1.0, 'CAD': 0.74, 'EUR': 1.09, 'GBP': 1.27, 'PLN': 0.25
        }
        
        # Define MCC codes
        self.mcc_codes = {
            '4816': 'Computer Network Services',
            '5045': 'Computers and Software',
            '5734': 'Software Stores',
            '7372': 'Programming Services',
            '5111': 'Office Supplies',
            '4814': 'Telecommunications',
            '5968': 'Subscription Services',
            '3000-3299': 'Airlines',
            '7011': 'Hotels and Lodging',
            '4121': 'Ground Transportation',
            '5811': 'Catering Services',
            '5812': 'Restaurants',
            '5399': 'General Merchandise'
        }
        
        # Regional merchants
        self.merchants = {
            'Cloud Services': {
                'US': ['AWS Amazon', 'Microsoft Azure US', 'Google Cloud US'],
                'Canada': ['AWS Canada', 'Microsoft Azure Canada', 'Google Cloud Canada'],
                'Europe': ['AWS EU', 'Microsoft Azure EU', 'Google Cloud EU'],
                'UK': ['AWS UK', 'Microsoft Azure UK', 'Google Cloud UK']
            },
            'Software': {
                'US': ['GitHub Enterprise', 'Atlassian US', 'JetBrains US', 'Adobe US'],
                'Canada': ['GitHub Canada', 'Atlassian Canada', 'JetBrains Canada', 'Adobe Canada'],
                'Europe': ['GitHub EU', 'Atlassian EU', 'JetBrains EU', 'Adobe EU'],
                'UK': ['GitHub UK', 'Atlassian UK', 'JetBrains UK', 'Adobe UK']
            },
            'Hardware': {
                'US': ['Dell US', 'Apple Business US', 'Lenovo US'],
                'Canada': ['Dell Canada', 'Apple Business Canada', 'Lenovo Canada'],
                'Europe': ['Dell EU', 'Apple Business EU', 'Lenovo EU'],
                'UK': ['Dell UK', 'Apple Business UK', 'Lenovo UK']
            },
            'Office Supplies': {
                'US': ['Staples US', 'Office Depot US', 'Amazon Business US'],
                'Canada': ['Staples Canada', 'Grand & Toy', 'Amazon Business Canada'],
                'Europe': ['Viking Direct EU', 'Amazon Business EU'],
                'UK': ['Viking Direct UK', 'Ryman Business', 'Amazon Business UK']
            },
            'Travel': {
                'US': ['United Airlines', 'American Airlines', 'Marriott US', 'Hilton US'],
                'Canada': ['Air Canada', 'WestJet', 'Fairmont Hotels', 'Delta Hotels'],
                'Europe': ['Lufthansa', 'Air France', 'NH Hotels', 'Accor Hotels'],
                'UK': ['British Airways', 'Virgin Atlantic', 'Premier Inn', 'IHG Hotels UK']
            },
            'Telecommunications': {
                'US': ['Verizon Business', 'AT&T Enterprise'],
                'Canada': ['Bell Business', 'Rogers Business'],
                'Europe': ['Deutsche Telekom', 'Orange Business'],
                'UK': ['BT Business', 'Vodafone UK']
            }
        }
        
        self.employees = self._generate_employee_profiles()
    
    def _generate_employee_profiles(self):
        employees = []
        roles = {
            'ENG-JR': {'limit': 5000, 'weight': 0.3},
            'ENG-SR': {'limit': 10000, 'weight': 0.3},
            'TECH-LEAD': {'limit': 20000, 'weight': 0.15},
            'ENG-MGR': {'limit': 50000, 'weight': 0.1},
            'DIR': {'limit': 100000, 'weight': 0.1},
            'VP': {'limit': 150000, 'weight': 0.04},
            'EXEC': {'limit': 500000, 'weight': 0.01}
        }
        
        departments = ['ENG', 'PROD', 'DEV', 'SEC', 'DATA', 'INFRA', 'QA', 'IT']
        employee_ids = random.sample(range(100000, 999999), self.num_employees)
        
        for emp_id in employee_ids:
            role = random.choices(list(roles.keys()), 
                                weights=[r['weight'] for r in roles.values()])[0]
            employees.append({
                'employee_id': f'EMP{emp_id}',
                'role': role,
                'department': random.choice(departments),
                'delegation_limit': roles[role]['limit'],
                'manager_id': None
            })
        
        # Assign managers
        for emp in employees:
            if emp['role'] != 'EXEC':
                potential_managers = [e for e in employees 
                                   if self._is_valid_manager(emp['role'], e['role'])]
                if potential_managers:
                    emp['manager_id'] = random.choice(potential_managers)['employee_id']
        
        return employees
    
    def _is_valid_manager(self, employee_role, manager_role):
        role_hierarchy = ['ENG-JR', 'ENG-SR', 'TECH-LEAD', 'ENG-MGR', 'DIR', 'VP', 'EXEC']
        return role_hierarchy.index(manager_role) > role_hierarchy.index(employee_role)
    
    def _generate_approval_timestamp(self, transaction_timestamp, is_anomalous=False):
        if not is_anomalous:
            min_delay = timedelta(minutes=5)
            max_delay = timedelta(hours=24)
            delay = min_delay + random.random() * (max_delay - min_delay)
            approval_timestamp = transaction_timestamp + delay
            
            while approval_timestamp.hour < 9 or approval_timestamp.hour >= 17 or approval_timestamp.weekday() >= 5:
                approval_timestamp = approval_timestamp + timedelta(hours=random.randint(1, 4))
        else:
            pattern_type = random.choice(['instant', 'delayed', 'off_hours', 'weekend'])
            
            if pattern_type == 'instant':
                delay = timedelta(seconds=random.randint(5, 55))
            elif pattern_type == 'delayed':
                delay = timedelta(hours=random.randint(48, 168))
            elif pattern_type == 'off_hours':
                delay = timedelta(hours=random.randint(1, 24))
                base_timestamp = transaction_timestamp + delay
                approval_timestamp = base_timestamp.replace(
                    hour=random.randint(0, 23) if random.random() < 0.5 else random.randint(0, 4),
                    minute=random.randint(0, 59)
                )
                return approval_timestamp
            elif pattern_type == 'weekend':
                delay = timedelta(hours=random.randint(1, 72))
                base_timestamp = transaction_timestamp + delay
                while base_timestamp.weekday() < 5:
                    base_timestamp += timedelta(days=1)
                approval_timestamp = base_timestamp.replace(
                    hour=random.randint(0, 23),
                    minute=random.randint(0, 59)
                )
                return approval_timestamp
            
            approval_timestamp = transaction_timestamp + delay
        
        return approval_timestamp
    
    def _generate_transaction_date(self):
        return self.start_date + timedelta(
            seconds=random.randint(0, int((self.end_date - self.start_date).total_seconds()))
        )
    
    def _generate_mcc_and_merchant(self):
        region = random.choice(['US', 'Canada', 'Europe', 'UK'])
        category = random.choice(list(self.merchants.keys()))
        merchant = random.choice(self.merchants[category][region])
        mcc = random.choice(list(self.mcc_codes.keys()))
        return mcc, merchant, category, region
    
    def _generate_normal_transaction(self):
        employee = random.choice(self.employees)
        mcc, merchant, category, region = self._generate_mcc_and_merchant()
        
        currency = {
            'US': 'USD',
            'Canada': 'CAD',
            'UK': 'GBP'
        }.get(region, random.choice(['EUR', 'PLN']))
        
        usd_amount = round(random.uniform(10, min(employee['delegation_limit'], 139514)), 2)
        amount = round(usd_amount / self.currencies[currency], 2)
        
        transaction_type = 'PURCHASE'
        if random.random() < 0.1:
            transaction_type = 'REFUND'
            amount = -amount
        
        date = self._generate_transaction_date()
        needs_approval = abs(usd_amount) > 100
        approver_id = None
        self_approved = False
        
        if needs_approval:
            if random.random() < 0.05:
                approver_id = employee['employee_id']
                self_approved = True
            else:
                approver_id = employee['manager_id']
        
        return {
            'transaction_id': f'TXN{random.randint(1000000000, 9999999999)}',
            'timestamp': date,
            'employee_id': employee['employee_id'],
            'employee_role': employee['role'],
            'department': employee['department'],
            'merchant': merchant,
            'merchant_region': region,
            'category': category,
            'mcc': mcc,
            'amount': amount,
            'currency': currency,
            'usd_amount': usd_amount,
            'transaction_type': transaction_type,
            'needs_approval': needs_approval,
            'approver_id': approver_id,
            'self_approved': self_approved,
            'is_anomaly': False,
            'anomaly_type': None,
            'approval_timestamp': self._generate_approval_timestamp(date, False) if needs_approval else None,
            'approval_pattern': 'normal' if needs_approval else None
        }
    
    def _generate_anomalous_transaction(self, base_transaction=None):
        if base_transaction is None:
            base_transaction = self._generate_normal_transaction()
        
        anomaly_types = ['duplicate', 'unusual_time', 'split_transaction', 
                        'unusual_merchant', 'self_approval_violation']
        anomaly_type = random.choice(anomaly_types)
        
        transaction = base_transaction.copy()
        transaction['transaction_id'] = f'TXN{random.randint(1000000000, 9999999999)}'
        transaction['is_anomaly'] = True
        transaction['anomaly_type'] = anomaly_type
        
        if transaction['needs_approval']:
            transaction['approval_timestamp'] = self._generate_approval_timestamp(
                transaction['timestamp'], is_anomalous=True)
            transaction['approval_pattern'] = 'anomalous'
        
        if anomaly_type == 'duplicate':
            transaction['timestamp'] += timedelta(minutes=random.randint(1, 60))
        elif anomaly_type == 'unusual_time':
            transaction['timestamp'] = transaction['timestamp'].replace(
                hour=random.randint(22, 23),
                minute=random.randint(0, 59)
            )
        elif anomaly_type == 'split_transaction':
            transaction['amount'] = transaction['amount'] / random.uniform(2, 4)
            transaction['usd_amount'] = transaction['usd_amount'] / random.uniform(2, 4)
            transaction['needs_approval'] = False
        elif anomaly_type == 'unusual_merchant':
            transaction['merchant'] = f'UNKNOWN-VENDOR-{random.randint(1000, 9999)}'
            transaction['category'] = 'Unknown'
            transaction['mcc'] = '7995'
        elif anomaly_type == 'self_approval_violation':
            if transaction['needs_approval']:
                transaction['approver_id'] = transaction['employee_id']
                transaction['self_approved'] = True
        
        return transaction
    
    def generate_dataset(self, num_transactions=25000, anomaly_ratio=0.05):
        transactions = []
        num_anomalies = int(num_transactions * anomaly_ratio)
        num_normal = num_transactions - num_anomalies
        
        for _ in range(num_normal):
            transactions.append(self._generate_normal_transaction())
        
        for _ in range(num_anomalies):
            if random.random() < 0.3:
                base_transaction = random.choice(transactions)
                transactions.append(self._generate_anomalous_transaction(base_transaction))
            else:
                transactions.append(self._generate_anomalous_transaction())
        
        df = pd.DataFrame(transactions)
        return df.sort_values('timestamp').reset_index(drop=True)