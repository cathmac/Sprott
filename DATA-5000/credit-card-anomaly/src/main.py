from transaction_generator import TransactionGenerator
from transaction_analysis import analyze_transactions, save_analysis_results
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

def main():
    # Create output directory for results
    output_dir = 'transaction_analysis_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    config = {
        'start_date': '2024-01-03',
        'end_date': '2024-12-06',
        'num_employees': 3000,
        'num_transactions': 25000,
        'anomaly_ratio': 0.05
    }
    
    print("Starting transaction data generation and analysis...")
    print(f"Configuration: {config}")
    
    # Generate dataset
    generator = TransactionGenerator(
        start_date=config['start_date'],
        end_date=config['end_date'],
        num_employees=config['num_employees']
    )
    
    df = generator.generate_dataset(
        num_transactions=config['num_transactions'],
        anomaly_ratio=config['anomaly_ratio']
    )
    
    # Save raw dataset
    raw_data_file = os.path.join(output_dir, 'credit_card_transactions.csv')
    df.to_csv(raw_data_file, index=False)
    print(f"\nRaw transaction data saved to: {raw_data_file}")
    
    # Perform analysis
    print("\nPerforming analysis...")
    fig = analyze_transactions(df)
    
    # Save visualization
    plot_file = os.path.join(output_dir, 'transaction_analysis_plots.png')
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Analysis plots saved to: {plot_file}")
    
    # Save analysis report
    report_file = os.path.join(output_dir, 'transaction_analysis_report.txt')
    save_analysis_results(df, report_file)
    print(f"Analysis report saved to: {report_file}")
    
    # Generate summary statistics for quick reference
    summary = {
        'Total Transactions': len(df),
        'Total Amount (USD)': df['usd_amount'].sum(),
        'Unique Employees': df['employee_id'].nunique(),
        'Normal Transactions': len(df[~df['is_anomaly']]),
        'Anomalous Transactions': len(df[df['is_anomaly']]),
        'Transactions Needing Approval': len(df[df['needs_approval']]),
        'Self-Approved Transactions': len(df[df['self_approved']])
    }
    
    summary_file = os.path.join(output_dir, 'summary_statistics.txt')
    with open(summary_file, 'w') as f:
        f.write("=== Transaction Analysis Summary ===\n\n")
        for key, value in summary.items():
            if isinstance(value, float):
                f.write(f"{key}: ${value:,.2f}\n")
            else:
                f.write(f"{key}: {value:,}\n")
    
    print(f"\nSummary statistics saved to: {summary_file}")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()