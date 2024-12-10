import pandas as pd
import matplotlib.pyplot as plt
import os
from transaction_generator import TransactionGenerator
from transaction_analysis import analyze_transactions, save_analysis_results
from data_preprocessing import CreditCardDataPreprocessor
from anomaly_detection import AnomalyDetectionModel
from sklearn.model_selection import train_test_split
import logging
import numpy as np

def main():
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MainPipeline")
    
    # Create output directory for results
    output_dir = 'transaction_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    config = {
        'start_date': '2024-01-03',
        'end_date': '2024-12-06',
        'num_employees': 3000,
        'num_transactions': 25000,
        'anomaly_ratio': 0.05,
        'doa_validator': {
            'role_limits': {
                'ENG-JR':    {'single': 5000,     'monthly': 15000},
                'ENG-SR':    {'single': 10000,    'monthly': 30000},
                'TECH-LEAD': {'single': 20000,    'monthly': 60000},
                'ENG-MGR':   {'single': 50000,    'monthly': 150000},
                'DIR':       {'single': 100000,   'monthly': 300000},
                'VP':        {'single': 150000,   'monthly': 450000},
                'EXEC':      {'single': 500000,   'monthly': 1500000},
                'CEO':       {'single': 1000000,  'monthly': 100000000},
                'BOARD':     {'single': float('inf'), 'monthly': float('inf')}
            },
            'business_hours_start': pd.to_datetime('06:00').time(),  # 6 AM EST
            'business_hours_end': pd.to_datetime('20:00').time()     # 8 PM EST
        },
        'risk_pattern_detector': {
            'severity_weights': {'low': 0.2, 'medium': 0.5, 'high': 1.0},
            'role_category_mapping': {
                'ENG-JR': ['Software', 'Hardware', 'Office Supplies'],
                'ENG-SR': ['Software', 'Hardware', 'Cloud Services', 'Office Supplies'],
                'TECH-LEAD': ['Software', 'Hardware', 'Cloud Services', 'Training'],
            },
            'role_merchant_mapping': {
                'ENG-JR': ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple'],
                'ENG-SR': ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple', 'JetBrains'],
                'TECH-LEAD': ['AWS', 'GitHub', 'Microsoft', 'Dell', 'Apple', 'JetBrains', 'Atlassian']
            },
            'max_history': 1000,
            'high_amount_threshold': 10000,  # Example threshold for high amount transactions
            'unusual_merchants': ['UnknownMerchant1', 'UnknownMerchant2'],  # Define as needed
            'unusual_categories': ['UnknownCategory1', 'UnknownCategory2']  # Define as needed
        },
        'threshold_optimizer': {
            'severity_weights': {'low': 0.2, 'medium': 0.5, 'high': 1.0}
        }
    }
    
    logger.info("Starting transaction data generation and analysis...")
    logger.info(f"Configuration: {config}")
    
    # Generate dataset
    generator = TransactionGenerator(
        start_date=config['start_date'],
        end_date=config['end_date'],
        num_employees=config['num_employees'],
        seed=42  # For reproducibility
    )
    
    df = generator.generate_dataset(
        num_transactions=config['num_transactions'],
        anomaly_ratio=config['anomaly_ratio']
    )
    
    if df is None:
        logger.error("DataFrame creation failed due to mismatched column lengths. Exiting pipeline.")
        exit(1)
    
    # Save raw dataset
    raw_data_file = os.path.join(output_dir, 'credit_card_transactions.csv')
    df.to_csv(raw_data_file, index=False)
    logger.info(f"Raw transaction data saved to: {raw_data_file}")
    
    # Perform analysis
    logger.info("Performing analysis...")
    fig = analyze_transactions(df)
    
    # Save visualization
    plot_file = os.path.join(output_dir, 'transaction_analysis_plots.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Analysis plots saved to: {plot_file}")
    
    # Save analysis report
    report_file = os.path.join(output_dir, 'transaction_analysis_report.txt')
    save_analysis_results(df, report_file)
    logger.info(f"Analysis report saved to: {report_file}")
    
    # Initialize preprocessor with config
    preprocessor = CreditCardDataPreprocessor(config=config)
    
    # Process data with validation and risk detection
    processed_df = preprocessor.process_data(
        input_file=raw_data_file,
        output_dir=output_dir
    )
    
    if processed_df is None:
        logger.error("Data preprocessing failed. Exiting pipeline.")
        exit(1)
    
    # Initialize and train model
    model = AnomalyDetectionModel(config=config)
    
    # Prepare features and target
    X, y = model.prepare_features_target(processed_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model.train_model(X_train, y_train)
    
    # Evaluate model
    results = model.evaluate_model(X_test, y_test)
    
    # Plot and save evaluation results
    model.plot_results(results)
    
    # Save model
    model.save_model(output_dir=output_dir)
    
    # Generate summary statistics for quick reference
    summary = {
        'Total Transactions': len(processed_df),
        'Total Amount (USD)': processed_df['usd_amount'].sum(),
        'Unique Employees': processed_df['employee_id'].nunique(),
        'Normal Transactions': len(processed_df[~processed_df['is_anomaly']]),
        'Anomalous Transactions': len(processed_df[processed_df['is_anomaly']]),
        'Transactions Needing Approval': len(processed_df[processed_df['needs_approval']]),
        'Self-Approved Transactions': len(processed_df[processed_df['self_approved']]),
        'High-Risk Transactions': len(processed_df[processed_df['is_high_risk']]),
        'Model ROC AUC': results['roc_auc'],
        # 'Model CV Mean' and 'Model CV Std' are not defined; remove or define them appropriately
    }
    
    # Adjusted summary to remove undefined 'Model CV Mean' and 'Model CV Std'
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    with open(summary_file, 'w') as f:
        f.write("=== Transaction Analysis Summary ===\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: ${value:,.2f}\n")
            elif isinstance(value, np.float64):
                f.write(f"{key}: ${value:,.2f}\n")
            else:
                f.write(f"{key}: {value:,}\n")
    logger.info(f"Summary statistics saved to: {summary_file}")
    logger.info("\nAnalysis complete!")

if __name__ == "__main__":
    main()
