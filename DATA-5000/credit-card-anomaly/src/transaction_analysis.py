import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys

def analyze_transactions(df):
    """Analyze transaction and approval patterns."""
    # Convert timestamps to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['approval_timestamp'] = pd.to_datetime(df['approval_timestamp'])
    
    # Basic Transaction Statistics
    print("\n=== Basic Transaction Statistics ===")
    print(f"Total Transactions: {len(df):,}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of Unique Employees: {df['employee_id'].nunique():,}")
    print(f"Total Amount (USD): ${df['usd_amount'].sum():,.2f}")
    print(f"Average Transaction Amount: ${df['usd_amount'].mean():,.2f}")
    
    # Transaction Distribution
    print("\n=== Transaction Distribution ===")
    print("\nTransaction Types:")
    print(df['transaction_type'].value_counts())
    print("\nCurrency Distribution:")
    print(df['currency'].value_counts())
    print("\nMerchant Regions:")
    print(df['merchant_region'].value_counts())
    
    # Anomaly Analysis
    print("\n=== Anomaly Analysis ===")
    print(f"Normal Transactions: {len(df[~df['is_anomaly']]):,}")
    print(f"Anomalous Transactions: {len(df[df['is_anomaly']]):,}")
    print(f"Anomaly Rate: {len(df[df['is_anomaly']]) / len(df):.2%}")
    
    if len(df[df['is_anomaly']]) > 0:
        print("\nAnomaly Types:")
        print(df[df['is_anomaly']]['anomaly_type'].value_counts())
    
    # Approval Analysis
    approved_txns = df[df['approval_timestamp'].notna()].copy()
    if len(approved_txns) > 0:
        approved_txns['approval_delay'] = (
            approved_txns['approval_timestamp'] - 
            approved_txns['timestamp']
        ).dt.total_seconds() / 3600
        
        print("\n=== Approval Analysis ===")
        print(f"Transactions Requiring Approval: {len(df[df['needs_approval']]):,}")
        print(f"Transactions with Approvals: {len(approved_txns):,}")
        print(f"Self-Approved Transactions: {len(df[df['self_approved']]):,}")
        
        print(f"\nApproval Timing:")
        print(f"Average Approval Delay: {approved_txns['approval_delay'].mean():.2f} hours")
        print(f"Median Approval Delay: {approved_txns['approval_delay'].median():.2f} hours")
        
        # Analyze approval patterns
        approved_txns['approval_hour'] = approved_txns['approval_timestamp'].dt.hour
        approved_txns['approval_day'] = approved_txns['approval_timestamp'].dt.day_name()
        
        print("\nApprovals by Day:")
        print(approved_txns['approval_day'].value_counts())
        
        # Off-hours approvals
        off_hours = approved_txns[
            (approved_txns['approval_hour'] >= 23) | 
            (approved_txns['approval_hour'] <= 4)
        ]
        print(f"\nOff-Hours Approvals (11PM-4AM): {len(off_hours):,}")
        
        # Weekend approvals
        weekend = approved_txns[approved_txns['approval_timestamp'].dt.dayofweek >= 5]
        print(f"Weekend Approvals: {len(weekend):,}")
    
    return create_visualizations(df, approved_txns)

def create_visualizations(df, approved_txns):
    """Create visualizations for transaction patterns."""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Transaction Analysis Dashboard', fontsize=16, y=0.95)
    
    # Create subplot grid
    gs = plt.GridSpec(3, 3, figure=fig)
    
    # 1. Transaction amounts by type
    ax1 = fig.add_subplot(gs[0, 0])
    sns.boxplot(data=df, x='transaction_type', y='usd_amount', ax=ax1)
    ax1.set_title('Transaction Amounts by Type')
    ax1.set_ylabel('Amount (USD)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Transactions by merchant region
    ax2 = fig.add_subplot(gs[0, 1])
    region_counts = df['merchant_region'].value_counts()
    sns.barplot(x=region_counts.index, y=region_counts.values, ax=ax2)
    ax2.set_title('Transactions by Region')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Currency distribution
    ax3 = fig.add_subplot(gs[0, 2])
    currency_counts = df['currency'].value_counts()
    sns.barplot(x=currency_counts.index, y=currency_counts.values, ax=ax3)
    ax3.set_title('Transactions by Currency')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Anomaly distribution
    ax4 = fig.add_subplot(gs[1, 0])
    anomaly_counts = df[df['is_anomaly']]['anomaly_type'].value_counts()
    sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values, ax=ax4)
    ax4.set_title('Distribution of Anomaly Types')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Transactions by department
    ax5 = fig.add_subplot(gs[1, 1])
    dept_counts = df['department'].value_counts()
    sns.barplot(x=dept_counts.index, y=dept_counts.values, ax=ax5)
    ax5.set_title('Transactions by Department')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Transactions by role
    ax6 = fig.add_subplot(gs[1, 2])
    role_counts = df['employee_role'].value_counts()
    sns.barplot(x=role_counts.index, y=role_counts.values, ax=ax6)
    ax6.set_title('Transactions by Employee Role')
    ax6.tick_params(axis='x', rotation=45)
    
    if len(approved_txns) > 0:
        # 7. Approval delays histogram
        ax7 = fig.add_subplot(gs[2, 0])
        sns.histplot(data=approved_txns, x='approval_delay', bins=50, ax=ax7)
        ax7.set_title('Distribution of Approval Delays')
        ax7.set_xlabel('Delay (hours)')
        
        # 8. Approvals by hour of day
        ax8 = fig.add_subplot(gs[2, 1])
        hour_counts = approved_txns['approval_hour'].value_counts().sort_index()
        sns.barplot(x=hour_counts.index, y=hour_counts.values, ax=ax8)
        ax8.set_title('Approvals by Hour of Day')
        ax8.set_xlabel('Hour')
        
        # 9. Approvals by day of week
        ax9 = fig.add_subplot(gs[2, 2])
        day_counts = approved_txns['approval_day'].value_counts()
        sns.barplot(x=day_counts.index, y=day_counts.values, ax=ax9)
        ax9.set_title('Approvals by Day of Week')
        ax9.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def save_analysis_results(df, output_file='transaction_analysis_report.txt'):
    """Save analysis results to a file."""
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        analyze_transactions(df)
        sys.stdout = original_stdout
        