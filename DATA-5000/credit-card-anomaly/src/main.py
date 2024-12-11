from anomaly_detection import AnomalyDetectionModel
from transaction_generator import TransactionGenerator
from transaction_analysis import analyze_transactions
from data_preprocessing import CreditCardDataPreprocessor
import os
import logging
from sklearn.model_selection import train_test_split


def main():
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MainPipeline")

    # Configuration
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'num_employees': 1000,
        'num_transactions': 20000,
        'anomaly_ratio': 0.02,
        'model_types': [
            'RandomForest', 'LogisticRegression', 'XGBoost'
        ]
    }

    output_dir = 'transaction_analysis_output'
    os.makedirs(output_dir, exist_ok=True)

    # Generate data
    generator = TransactionGenerator(
        start_date=config['start_date'],
        end_date=config['end_date'],
        num_employees=config['num_employees']
    )
    df = generator.generate_dataset(num_transactions=config['num_transactions'], anomaly_ratio=config['anomaly_ratio'])
    raw_data_file = os.path.join(output_dir, 'credit_card_transactions.csv')
    df.to_csv(raw_data_file, index=False)

    # Preprocess data
    preprocessor = CreditCardDataPreprocessor(config=config)
    processed_df = preprocessor.process_data(raw_data_file, output_dir)
    X, y = processed_df.drop(columns=['is_anomaly']), processed_df['is_anomaly']

    # Check for missing values
    print(f"Missing values in X:\n{X.isnull().sum()}")
    print(f"Missing values in y:\n{y.isnull().sum()}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Evaluate each model type
    for model_type in config['model_types']:
        logger.info(f"\nEvaluating {model_type}...")
        model = AnomalyDetectionModel(config=config, model_type=model_type)
        model.train_model(X_train, y_train)
        results = model.evaluate_model(X_test, y_test)
        if results:
            model.plot_results(results)
            model.save_model(output_dir)


if __name__ == "__main__":
    main()
