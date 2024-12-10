import pytz
import pandas as pd
from datetime import datetime, time

class DOAValidator:
    def __init__(self):
        # Role-based limits
        self.role_limits = {
            'ENG-JR':    {'single': 5000,     'monthly': 15000},
            'ENG-SR':    {'single': 10000,    'monthly': 30000},
            'TECH-LEAD': {'single': 20000,    'monthly': 60000},
            'ENG-MGR':   {'single': 50000,    'monthly': 150000},
            'DIR':       {'single': 100000,   'monthly': 300000},
            'VP':        {'single': 150000,   'monthly': 450000},
            'EXEC':      {'single': 500000,   'monthly': 1500000},
            'CEO':       {'single': 1000000,  'monthly': 100000000},
            'BOARD':     {'single': float('inf'), 'monthly': float('inf')}  # Unlimited
        }
        
        # Role hierarchy (optional usage)
        self.role_hierarchy = [
            'ENG-JR',
            'ENG-SR',
            'TECH-LEAD',
            'ENG-MGR',
            'DIR',
            'VP',
            'EXEC',
            'CEO',
            'BOARD'
        ]
        
        # Define business hours in Eastern Time
        self.business_hours_start = time(6, 0)  # 6 AM EST
        self.business_hours_end = time(20, 0)   # 8 PM EST
        self.est_tz = pytz.timezone('US/Eastern')

    def _check_role_limit(self, transaction):
        """Check if transaction amount is within the single-transaction limit for the role."""
        role = transaction['employee_role']
        amount = transaction['usd_amount']

        # If role not found, default to True; BOARD is unlimited
        if role not in self.role_limits:
            return True
        if role == 'BOARD':
            return True

        return amount <= self.role_limits[role]['single']

    def _check_approval_requirements(self, transaction):
        """Check various approval requirements based on transaction amount and role."""
        violations = []
        amount = transaction['usd_amount']
        role = transaction['employee_role']

        # Board members have no additional approval requirements
        if role == 'BOARD':
            return violations

        # Require approval for transactions over $100
        if amount > 100 and not transaction.get('needs_approval', False):
            violations.append('Transaction over $100 requires approval')

        # Check self-approval
        if transaction.get('self_approved', False):
            violations.append('Self-approval is not permitted')

        # Two-level approval required for transactions > $25,000
        if amount > 25000 and transaction.get('secondary_approval') != True:
            violations.append('Transactions over $25,000 require two-level approval')

        # Transactions > $100,000 require VP or above approval
        if amount > 100000:
            approver_role = transaction.get('approver_role')
            if not self._is_vp_or_above(approver_role):
                violations.append('Transactions over $100,000 require VP level or above approval')

        # Transactions > $1,000,000 require Board approval (unless role is BOARD)
        if amount > 1000000 and role != 'BOARD':
            if not transaction.get('board_approval', False):
                violations.append('Transactions over $1,000,000 require Board approval')

        return violations

    def _is_vp_or_above(self, role):
        """Check if the given role is VP level or above."""
        if role is None:
            return False
        vp_roles = ['VP', 'EXEC', 'CEO', 'BOARD']
        return role in vp_roles

    def _check_monthly_limit(self, employee_id, role, month_total):
        """
        Check if monthly spending is within role limits.
        Adjust logic as needed when monthly totals are calculated.
        """
        if role == 'BOARD':
            # Board members have no monthly limit
            return True

        if role not in self.role_limits:
            return True  # If role is unknown, default to True or handle separately

        monthly_limit = self.role_limits[role]['monthly']
        return month_total <= monthly_limit

    def _check_timing_rules(self, tx_time):
        """Check if the transaction time falls within business hours (EST)."""
        violations = []
        if not (self.business_hours_start <= tx_time.time() <= self.business_hours_end):
            violations.append('Transaction outside business hours')
        return violations

    def _is_potential_split_transaction(self, transaction):
        """
        Placeholder for detecting split transactions.
        Implement logic based on transaction history if needed.
        """
        return False

    def validate_transaction(self, transaction):
        """
        Validate a transaction against DOA policy rules.
        Returns (is_valid, violations)
        """
        violations = []

        # Check role limit
        if not self._check_role_limit(transaction):
            violations.append('Exceeds role-based transaction limit')

        # Check approval requirements
        approval_violations = self._check_approval_requirements(transaction)
        violations.extend(approval_violations)

        # Check timing rules for non-BOARD roles
        role = transaction['employee_role']
        if role != 'BOARD' and 'timestamp' in transaction:
            # Convert timestamp to EST
            timestamp_utc = pd.to_datetime(transaction['timestamp'], utc=True)
            tx_time_est = timestamp_utc.astimezone(self.est_tz)
            timing_violations = self._check_timing_rules(tx_time_est)
            violations.extend(timing_violations)

        # Check for potential split transactions
        if self._is_potential_split_transaction(transaction):
            violations.append('Potential split transaction detected')

        return len(violations) == 0, violations
