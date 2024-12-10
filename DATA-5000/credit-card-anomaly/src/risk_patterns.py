from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class RiskPatternDetector:
    def __init__(self):
        self.transaction_history = []
        self.max_history = 1000
        
        # Time-based patterns
        self.time_patterns = {
            'late_night': {
                'check': lambda tx: 0 <= tx['hour'] <= 4,
                'severity': 'high',
                'message': "Transaction between midnight and 4 AM"
            },
            'weekend': {
                'check': lambda tx: tx['is_weekend'],
                'severity': 'medium',
                'message': "Weekend transaction"
            },
            'holiday': {
                'check': self._is_holiday,
                'severity': 'medium',
                'message': "Transaction on holiday"
            },
            'end_of_month': {
                'check': lambda tx: pd.to_datetime(tx['timestamp']).day >= 28,
                'severity': 'low',
                'message': "End of month transaction"
            }
        }
        
        # Amount-based patterns
        self.amount_patterns = {
            'large_amount': {
                'check': lambda tx: float(tx['amount']) > 10000,
                'severity': 'high',
                'message': "Large transaction amount"
            },
            'threshold_amount': {
                'check': self._is_threshold_amount,
                'severity': 'high',
                'message': "Amount just below approval threshold"
            },
            'round_amount': {
                'check': lambda tx: float(tx['amount']) % 100 == 0,
                'severity': 'low',
                'message': "Round number amount"
            },
            'repeated_amount': {
                'check': self._check_repeated_amount,
                'severity': 'high',
                'message': "Repeated amount in short time period"
            }
        }
        
        # Approval-based patterns
        self.approval_patterns = {
            'self_approval': {
                'check': lambda tx: tx.get('self_approved', False),
                'severity': 'high',
                'message': "Self-approved transaction"
            },
            'quick_approval': {
                'check': lambda tx: tx.get('approval_delay_hours', 0) < 0.1,
                'severity': 'medium',
                'message': "Unusually quick approval"
            },
            'delayed_approval': {
                'check': lambda tx: tx.get('approval_delay_hours', 0) > 48,
                'severity': 'medium',
                'message': "Significantly delayed approval"
            },
            'missing_approval': {
                'check': lambda tx: tx.get('needs_approval', False) and not tx.get('approver_id'),
                'severity': 'high',
                'message': "Missing required approval"
            }
        }
        
        # Merchant-based patterns
        self.merchant_patterns = {
            'new_merchant': {
                'check': self._check_new_merchant,
                'severity': 'medium',
                'message': "Transaction with new merchant"
            },
            'unusual_category': {
                'check': self._check_unusual_category,
                'severity': 'medium',
                'message': "Unusual merchant category for role"
            },
            'high_risk_merchant': {
                'check': self._check_high_risk_merchant,
                'severity': 'high',
                'message': "Transaction with high-risk merchant"
            },
            'merchant_location_mismatch': {
                'check': self._check_merchant_location,
                'severity': 'high',
                'message': "Merchant location inconsistency"
            }
        }
        
        # Behavior-based patterns
        self.behavior_patterns = {
            'unusual_frequency': {
                'check': self._check_unusual_frequency,
                'severity': 'medium',
                'message': "Unusual transaction frequency"
            },
            'pattern_break': {
                'check': self._check_pattern_break,
                'severity': 'medium',
                'message': "Break from normal transaction pattern"
            },
            'role_mismatch': {
                'check': self._check_role_merchant_mismatch,
                'severity': 'high',
                'message': "Unusual merchant for employee role"
            }
        }

    def _is_holiday(self, tx):
        holidays = [
            '2024-01-01', '2024-12-25', '2024-12-26',
            '2024-07-04', '2024-11-28', '2024-11-29',
            '2024-05-27', '2024-09-02', '2024-10-14',
            '2024-02-19', '2024-01-15'
        ]
        tx_date = pd.to_datetime(tx['timestamp']).strftime('%Y-%m-%d')
        return tx_date in holidays

    def _is_threshold_amount(self, tx):
        thresholds = [100, 1000, 5000, 10000, 25000]
        amount = float(tx['amount'])
        return any(0.90 * t <= amount <= 0.99 * t for t in thresholds)

    def _check_repeated_amount(self, tx):
        if not self.transaction_history:
            return False
        
        recent = pd.DataFrame(self.transaction_history[-20:])
        amount = float(tx['amount'])
        return len(recent[
            (recent['employee_id'] == tx['employee_id']) &
            (np.isclose(recent['amount'], amount, atol=0.01)) &
            ((pd.to_datetime(tx['timestamp']) - pd.to_datetime(recent['timestamp'])).dt.total_seconds() < 86400)
        ]) > 0

    def _check_new_merchant(self, tx):
        if not self.transaction_history:
            return True
            
        recent = pd.DataFrame(self.transaction_history)
        employee_merchants = recent[recent['employee_id'] == tx['employee_id']]['merchant'].unique()
        return tx['merchant'] not in employee_merchants

    def _check_unusual_category(self, tx):
        role_category_mapping = {
            'ENG-JR': ['Software', 'Hardware', 'Office Supplies'],
            'ENG-SR': ['Software', 'Hardware', 'Cloud Services', 'Office Supplies'],
            'TECH-LEAD': ['Software', 'Hardware', 'Cloud Services', 'Training'],
        }

        if tx['employee_role'] in ['ENG-MGR', 'DIR', 'VP', 'EXEC', 'CEO', 'BOARD']:
            return False
            
        allowed_categories = role_category_mapping.get(tx['employee_role'], [])
        return (tx['category'] not in allowed_categories) and ('All' not in allowed_categories)

    def _check_high_risk_merchant(self, tx):
        high_risk_categories = [
            'Cryptocurrency', 'Gaming', 'Adult Entertainment',
            'Foreign Exchange', 'Wire Transfer Services'
        ]
        return tx['category'] in high_risk_categories

    def _check_merchant_location(self, tx):
        if 'merchant_country' not in tx or 'transaction_country' not in tx:
            return False
        return tx['merchant_country'] != tx['transaction_country']

    def _check_unusual_frequency(self, tx):
        if not self.transaction_history:
            return False
            
        recent = pd.DataFrame(self.transaction_history[-50:])
        recent['timestamp'] = pd.to_datetime(recent['timestamp'])
        tx_time = pd.to_datetime(tx['timestamp'])
        
        count_24h = len(recent[
            (recent['employee_id'] == tx['employee_id']) &
            (recent['timestamp'] >= tx_time - timedelta(hours=24))
        ])
        
        role_thresholds = {
            'ENG-JR': 3,
            'ENG-SR': 4,
            'TECH-LEAD': 5,
            'ENG-MGR': 7,
            'DIR': 10,
            'VP': 15,
            'EXEC': 20
        }
        
        return count_24h > role_thresholds.get(tx['employee_role'], 5)

    def _check_pattern_break(self, tx):
        if not self.transaction_history:
            return False
            
        recent = pd.DataFrame(self.transaction_history[-100:])
        employee_tx = recent[recent['employee_id'] == tx['employee_id']]
        
        if employee_tx.empty:
            return False
        
        avg_amount = employee_tx['amount'].mean()
        current_amount = float(tx['amount'])
        
        return abs(current_amount - avg_amount) > 2 * avg_amount

    def _check_role_merchant_mismatch(self, tx):
        role_merchant_mapping = {
            'ENG-JR': ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple'],
            'ENG-SR': ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple', 'JetBrains'],
            'TECH-LEAD': ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple', 'JetBrains', 'Atlassian']
        }
        
        if tx['employee_role'] in ['ENG-MGR', 'DIR', 'VP', 'EXEC', 'CEO', 'BOARD']:
            return False
            
        allowed_merchants = role_merchant_mapping.get(tx['employee_role'], [])
        return not any(m in tx['merchant'] for m in allowed_merchants)

    def evaluate_transaction(self, transaction):
        """
        Evaluate all risk patterns for a given transaction.
        We must ensure 'hour' and 'is_weekend' are defined before pattern checks.
        """
        # Ensure timestamp is parsed
        tx_time = pd.to_datetime(transaction['timestamp'])
        
        # Define 'hour' and 'is_weekend' before checking patterns
        transaction['hour'] = tx_time.hour
        transaction['is_weekend'] = tx_time.weekday() >= 5

        risk_patterns = []
        total_risk_score = 0
        severity_weights = {'low': 0.2, 'medium': 0.5, 'high': 1.0}
        
        pattern_groups = [
            self.time_patterns,
            self.amount_patterns,
            self.approval_patterns,
            self.merchant_patterns,
            self.behavior_patterns
        ]

        for patterns in pattern_groups:
            for pattern_name, pattern in patterns.items():
                try:
                    if pattern['check'](transaction):
                        risk_score = severity_weights[pattern['severity']]
                        risk_patterns.append({
                            'pattern': pattern_name,
                            'message': pattern['message'],
                            'severity': pattern['severity'],
                            'risk_score': risk_score
                        })
                        total_risk_score += risk_score
                except Exception as e:
                    print(f"Error checking {pattern_name}: {str(e)}")
        
        self._update_history(transaction)
        
        return risk_patterns, min(1.0, total_risk_score / 3.0)

    def _update_history(self, transaction):
        self.transaction_history.append(transaction)
        if len(self.transaction_history) > self.max_history:
            self.transaction_history.pop(0)

def main():
    detector = RiskPatternDetector()
    
    transaction = {
        'timestamp': '2024-03-15T23:45:00',
        'employee_id': 'EMP123456',
        'employee_role': 'ENG-SR',
        'amount': 9999.00,
        'merchant': 'NEW-VENDOR-XYZ',
        'category': 'Software',
        'needs_approval': True,
        'approver_id': 'EMP123456',
        'self_approved': True,
        'approval_delay_hours': 0.05
    }
    
    risk_patterns, risk_score = detector.evaluate_transaction(transaction)
    
    print(f"\nRisk Score: {risk_score:.2f}")
    print("Detected Risk Patterns:")
    for pattern in risk_patterns:
        print(f"- {pattern['message']} (Severity: {pattern['severity']})")

if __name__ == "__main__":
    main()
