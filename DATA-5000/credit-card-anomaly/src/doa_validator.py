import pandas as pd
import pytz
import logging

class DOAValidator:
    def __init__(self, config=None):
        """
        Initializes the DOAValidator with role-based limits and business hours.
        
        Parameters:
        - config (dict, optional): Configuration dictionary for role limits and business hours.
        """
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)
        
        # Role-based limits
        self.role_limits = config.get('role_limits') if config and 'role_limits' in config else {
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
        
        # Define business hours in Eastern Time
        self.business_hours_start = config.get('business_hours_start') if config and 'business_hours_start' in config else pd.to_datetime('06:00').time()  # 6 AM EST
        self.business_hours_end = config.get('business_hours_end') if config and 'business_hours_end' in config else pd.to_datetime('20:00').time()   # 8 PM EST
        self.est_tz = pytz.timezone('US/Eastern')
    
    def _check_role_limit(self, transaction):
        """Check if transaction amount is within the single-transaction limit for the role."""
        role = transaction.get('employee_role')
        amount = transaction.get('usd_amount', 0)
        
        # If role not found, default to True; BOARD is unlimited
        if role not in self.role_limits:
            self.logger.warning(f"Role '{role}' not found in role limits. Defaulting to True.")
            return True
        if role == 'BOARD':
            return True

        return amount <= self.role_limits[role]['single']
    
    def _check_approval_requirements(self, transaction):
        """Check various approval requirements based on transaction amount and role."""
        violations = []
        amount = transaction.get('usd_amount', 0)
        role = transaction.get('employee_role')
        
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
        if amount > 25000 and not transaction.get('secondary_approval', False):
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
        
        Parameters:
        - employee_id (str): The ID of the employee.
        - role (str): The role of the employee.
        - month_total (float): The total amount spent in the month.
        
        Returns:
        - bool: True if within limit, False otherwise.
        """
        if role == 'BOARD':
            # Board members have no monthly limit
            return True

        if role not in self.role_limits:
            self.logger.warning(f"Role '{role}' not found in role limits. Defaulting to True.")
            return True  # If role is unknown, default to True or handle separately

        monthly_limit = self.role_limits[role]['monthly']
        return month_total <= monthly_limit
    
    def _check_timing_rules(self, tx_time):
        """Check if the transaction time falls within business hours (EST)."""
        violations = []
        if not (self.business_hours_start <= tx_time.time() <= self.business_hours_end):
            violations.append('Transaction outside business hours')
        return violations
    
    def _is_potential_split_transaction(self, transaction, transaction_history=None):
        """
        Detect potential split transactions based on historical data.
        
        Parameters:
        - transaction (dict): The current transaction.
        - transaction_history (pd.DataFrame, optional): Historical transactions.
        
        Returns:
        - bool: True if a split transaction is detected, False otherwise.
        """
        if transaction_history is None or transaction_history.empty:
            return False

        employee_id = transaction.get('employee_id')
        amount = float(transaction.get('usd_amount', 0))
        timestamp = pd.to_datetime(transaction.get('timestamp'), utc=True)
        threshold_time = timestamp - pd.Timedelta(minutes=30)
        
        # Filter transactions from the same employee within the last 30 minutes
        recent_transactions = transaction_history[
            (transaction_history['employee_id'] == employee_id) &
            (transaction_history['timestamp'] >= threshold_time) &
            (transaction_history['timestamp'] <= timestamp)
        ]
        
        # Define criteria for split transactions
        if len(recent_transactions) >= 3 and amount < 100:
            return True
        
        return False
    
    def validate_transaction(self, transaction, transaction_history=None):
        """
        Validate a transaction against DOA policy rules.
        
        Parameters:
        - transaction (dict): The transaction data.
        - transaction_history (pd.DataFrame, optional): Historical transactions for split detection.
        
        Returns:
        - tuple: (is_valid (bool), violations (list of str))
        """
        violations = []
        
        try:
            # Check role limit
            if not self._check_role_limit(transaction):
                violations.append('Exceeds role-based transaction limit')
            
            # Check approval requirements
            approval_violations = self._check_approval_requirements(transaction)
            violations.extend(approval_violations)
            
            # Check timing rules for non-BOARD roles
            role = transaction.get('employee_role')
            timestamp = transaction.get('timestamp')
            if role and role != 'BOARD' and timestamp:
                # Convert timestamp to EST
                timestamp_utc = pd.to_datetime(timestamp, utc=True)
                tx_time_est = timestamp_utc.astimezone(self.est_tz)
                timing_violations = self._check_timing_rules(tx_time_est)
                violations.extend(timing_violations)
            
            # Check for potential split transactions
            if self._is_potential_split_transaction(transaction, transaction_history):
                violations.append('Potential split transaction detected')
        
        except Exception as e:
            self.logger.error(f'Error during validation: {str(e)}')
            violations.append(f'Error during validation: {str(e)}')
        
        is_valid = len(violations) == 0
        return is_valid, violations

