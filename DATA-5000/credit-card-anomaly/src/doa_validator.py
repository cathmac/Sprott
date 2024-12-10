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
        
        # Define role hierarchy
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
        
        # Define business hours (EST)
        self.business_hours_start = time(6, 0)  # 6 AM
        self.business_hours_end = time(20, 0)   # 8 PM
        self.est_tz = pytz.timezone('US/Eastern')

    def _check_role_limit(self, transaction):
        """Check if transaction amount is within role limit."""
        role = transaction['employee_role']
        amount = transaction['usd_amount']
        
        # Board members have unlimited authority
        if role == 'BOARD':
            return True
            
        return amount <= self.role_limits[role]['single']

    def _check_approval_requirements(self, transaction):
        violations = []
        amount = transaction['usd_amount']
        role = transaction['employee_role']
        
        # Board members don't need additional approvals
        if role == 'BOARD':
            return violations
            
        # Basic approval threshold
        if amount > 100 and not transaction['needs_approval']:
            violations.append('Transaction over $100 requires approval')
        
        # Self-approval check (even CEO and Board need dual control)
        if transaction['self_approved']:
            violations.append('Self-approval is not permitted')
        
        # High-value transaction checks
        if amount > 25000 and transaction.get('secondary_approval') != True:
            violations.append('Transactions over $25,000 require two-level approval')
            
        # Special approval requirements for very large transactions
        if amount > 100000:
            if not self._is_vp_or_above(transaction.get('approver_role')):
                violations.append('Transactions over $100,000 require VP level or above approval')
            
        if amount > 1000000 and role != 'BOARD':
            if not transaction.get('board_approval'):
                violations.append('Transactions over $1,000,000 require Board approval')
        
        return violations

    def _is_vp_or_above(self, role):
        """Check if role is VP level or above."""
        vp_roles = ['VP', 'EXEC', 'CEO', 'BOARD']
        return role in vp_roles

    def _check_monthly_limit(self, employee_id, role, month_total):
        """Check if monthly spending is within limits."""
        if role == 'BOARD':  # Board members have unlimited monthly limit
            return True
            
        monthly_limit = self.role_limits[role]['monthly']
        return month_total <= monthly_limit

    def validate_transaction(self, transaction):
        """
        Validates a transaction against DOA policy rules.
        Returns (is_valid, violations)
        """
        violations = []
        
        # Convert timestamp to EST
        import pandas as pd
        
        # Check role-based limits
        if not self._check_role_limit(transaction):
            violations.append('Exceeds role-based transaction limit')
            
        # Check approval requirements
        approval_violations = self._check_approval_requirements(transaction)
        violations.extend(approval_violations)
        
        # Check timing rules (excluding Board members)
        if transaction['employee_role'] != 'BOARD':
            timing_violations = self._check_timing_rules(tx_time)
            violations.extend(timing_violations)
        
        # Check for split transactions
        if self._is_potential_split_transaction(transaction):
            violations.append('Potential split transaction detected')
        
        return len(violations) == 0, violations