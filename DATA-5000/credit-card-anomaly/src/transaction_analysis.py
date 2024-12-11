import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def analyze_transactions(df):
    """
    Performs basic transaction analysis and generates visualizations.
    
    Parameters:
    - df (pd.DataFrame): Processed transaction DataFrame.
    
    Returns:
    - matplotlib.figure.Figure: The generated plot figure.
    """
    logger = logging.getLogger("TransactionAnalysis")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    
    logger.info("Starting transaction analysis...")
    
    # Example analyses:
    # 1. Basic statistics
    # 2. Distribution of transaction types
    # 3. Currency distribution
    # 4. Merchant regions distribution
    # 5. Anomaly analysis
    # 6. Approval analysis
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Transaction Analysis', fontsize=20)
    
    # 1. Basic Transaction Statistics
    axes[0, 0].text(0.1, 0.9, f"Total Transactions: {len(df):,}", fontsize=12)
    axes[0, 0].text(0.1, 0.8, f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}", fontsize=12)
    axes[0, 0].text(0.1, 0.7, f"Number of Unique Employees: {df['employee_id'].nunique():,}", fontsize=12)
    axes[0, 0].text(0.1, 0.6, f"Total Amount (USD): ${df['usd_amount'].sum():,.2f}", fontsize=12)
    axes[0, 0].text(0.1, 0.5, f"Average Transaction Amount: ${df['usd_amount'].mean():.2f}", fontsize=12)
    axes[0, 0].axis('off')
    
    # 2. Transaction Types Distribution
    sns.countplot(ax=axes[0, 1], x='transaction_type', data=df, order=df['transaction_type'].value_counts().index)
    axes[0, 1].set_title('Transaction Types')
    axes[0, 1].set_xlabel('Transaction Type')
    axes[0, 1].set_ylabel('Count')
    
    # 3. Currency Distribution
    sns.countplot(ax=axes[1, 0], x='currency', data=df, order=df['currency'].value_counts().index)
    axes[1, 0].set_title('Currency Distribution')
    axes[1, 0].set_xlabel('Currency')
    axes[1, 0].set_ylabel('Count')
    
    # 4. Merchant Regions Distribution
    sns.countplot(ax=axes[1, 1], x='merchant_region', data=df, order=df['merchant_region'].value_counts().index)
    axes[1, 1].set_title('Merchant Regions')
    axes[1, 1].set_xlabel('Merchant Region')
    axes[1, 1].set_ylabel('Count')
    
    # 5. Anomaly Analysis
    sns.countplot(ax=axes[2, 0], x='is_anomaly', data=df)
    axes[2, 0].set_title('Anomaly Analysis')
    axes[2, 0].set_xlabel('Is Anomaly')
    axes[2, 0].set_ylabel('Count')
    axes[2, 0].set_xticks([0, 1])  # Ensure fixed number of ticks
    axes[2, 0].set_xticklabels(['Normal', 'Anomalous'])
    
    # 6. Approval Analysis
    approval_counts = df['needs_approval'].value_counts()
    sns.barplot(ax=axes[2, 1], x=approval_counts.index.astype(str), y=approval_counts.values)
    axes[2, 1].set_title('Transactions Requiring Approval')
    axes[2, 1].set_xlabel('Requires Approval')
    axes[2, 1].set_ylabel('Count')
    axes[2, 1].set_xticks([0, 1])  # Ensure fixed number of ticks
    axes[2, 1].set_xticklabels(['No', 'Yes'])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    logger.info("Transaction analysis completed.")
    return fig

def save_analysis_results(df, report_file):
    """
    Saves analysis results to a text file.
    
    Parameters:
    - df (pd.DataFrame): Processed transaction DataFrame.
    - report_file (str): Path to the report text file.
    """
    logger = logging.getLogger("TransactionAnalysis")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    
    logger.info("Saving analysis results...")
    
    with open(report_file, 'w') as f:
        f.write("=== Basic Transaction Statistics ===\n")
        f.write(f"Total Transactions: {len(df):,}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n")
        f.write(f"Number of Unique Employees: {df['employee_id'].nunique():,}\n")
        f.write(f"Total Amount (USD): ${df['usd_amount'].sum():,.2f}\n")
        f.write(f"Average Transaction Amount: ${df['usd_amount'].mean():.2f}\n\n")
        
        f.write("=== Transaction Distribution ===\n\n")
        
        f.write("Transaction Types:\n")
        f.write(f"{df['transaction_type'].value_counts()}\n\n")
        
        f.write("Currency Distribution:\n")
        f.write(f"{df['currency'].value_counts()}\n\n")
        
        f.write("Merchant Regions:\n")
        f.write(f"{df['merchant_region'].value_counts()}\n\n")
        
        f.write("=== Anomaly Analysis ===\n")
        normal = len(df[~df['is_anomaly']])
        anomalous = len(df[df['is_anomaly']])
        anomaly_rate = (anomalous / len(df)) * 100
        f.write(f"Normal Transactions: {normal:,}\n")
        f.write(f"Anomalous Transactions: {anomalous:,}\n")
        f.write(f"Anomaly Rate: {anomaly_rate:.2f}%\n\n")
        
        f.write("Anomaly Types:\n")
        f.write(f"{df['anomaly_type'].value_counts(dropna=True)}\n\n")
        
        f.write("=== Approval Analysis ===\n")
        approval_required = len(df[df['needs_approval']])
        approvals = len(df[df['approver_id'] != 'NONE'])
        self_approved = len(df[df['self_approved']])
        f.write(f"Transactions Requiring Approval: {approval_required:,}\n")
        f.write(f"Transactions with Approvals: {approvals:,}\n")
        f.write(f"Self-Approved Transactions: {self_approved:,}\n")
    
    logger.info(f"Analysis results saved to: {report_file}")

