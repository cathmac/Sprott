#!/usr/bin/env python3

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
import cProfile
import pstats


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
        'num_transactions': 50000,
        'anomaly_ratio': 0.02,
        'model_types': [
            'IsolationForest', 'OneClassSVM', 'LOF', 'DBSCAN', 'EllipticEnvelope',
            'KMeans', 'RandomForest', 'LogisticRegression', 'GaussianMixture'
        ],
        'imbalance_method': 'undersample',
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
        logger.error("DataFrame creation failed. Exiting pipeline.")
        exit(1)
    
    # Save raw dataset
    raw_data_file = os.path.join(output_dir, 'credit_card_transactions.csv')
    df.to_csv(raw_data_file, index=False)
    logger.info(f"Raw transaction data saved to: {raw_data_file}")
    
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
    
    # Drop datetime columns
    datetime_columns = processed_df.select_dtypes(include=['datetime64']).columns
    logger.info(f"Dropping datetime columns: {datetime_columns}")
    processed_df = processed_df.drop(columns=datetime_columns, errors='ignore')
    
    # Prepare features and target
    X, y = processed_df.drop(columns=['is_anomaly']), processed_df['is_anomaly']
    
    # Drop non-numeric columns explicitly
    X = X.select_dtypes(include=[np.number])
    
    # Handle class imbalance
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Iterate through different model types
    for model_type in config['model_types']:
        logger.info(f"\n=== Evaluating {model_type} ===")
        
        # Initialize and train model
        model = AnomalyDetectionModel(config=config, model_type=model_type)
        model.train_model(X_train, y_train)
        
        # Evaluate model
        results = model.evaluate_model(X_test, y_test)
        
        # Save evaluation report
        report_file = os.path.join(output_dir, f"{model_type}_evaluation_report.txt")
        with open(report_file, 'w') as f:
            f.write(f"Model: {model_type}\n")
            f.write(f"ROC AUC: {results['roc_auc']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1 Score: {results['f1_score']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(results['classification_report'])
        
        logger.info(f"Evaluation report saved to: {report_file}")
        
        # Plot and save evaluation results
        model.plot_results(results)
        
        # Save model
        model.save_model(output_dir=output_dir)

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    
    main()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)
