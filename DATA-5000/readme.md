# Credit Card Transaction Analysis Framework

This Python framework analyzes corporate credit card transactions, detects anomalies, and ensures compliance with the delegation of authority (DOA) policy. It performs end-to-end processing, from synthetic transaction generation to validation, anomaly detection, and risk pattern analysis.

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Running the Framework](#running-the-framework)  
4. [Delegation of Authority Validation](#delegation-of-authority-validation)  
5. [Colab Notebook Instructions](#colab-notebook-instructions)  
6. [Files Overview](#files-overview)  

---

## 1. Prerequisites

- **Python 3.x**  
- Required libraries listed in `requirements.txt` (install via `pip`).  

## 2. Installation

To install dependencies, run:

```bash
pip install -r requirements.txt
```

## 3. Running the Framework

The main entry point for the pipeline is `2main.py`. To execute the pipeline, run:

```bash
python 2main.py
```

### What the Framework Does:

1. **Generates Synthetic Transactions**  
   - Uses `transaction_generator.py` to create random transactions.
   
2. **Preprocesses Data**  
   - Cleans data and validates transactions using the DOA policy via `data_preprocessing.py` and `doa_validator.py`.

3. **Detects Anomalies**  
   - Applies various anomaly detection models (`IsolationForest`, `DBSCAN`, etc.) defined in `anomaly_detection.py`.

4. **Analyzes Results**  
   - Generates visualizations and saves analysis reports using `transaction_analysis.py`.

### Configuration

You can modify the configuration settings in `2main.py` under the `config` dictionary:

```python
config = {
    'start_date': '2024-01-03',
    'end_date': '2024-12-06',
    'num_employees': 3000,
    'num_transactions': 50000,
    'anomaly_ratio': 0.02,
    'model_types': ['IsolationForest', 'OneClassSVM', 'LOF', 'DBSCAN'],
    'imbalance_method': 'undersample',
}
```

### Output

Results will be saved in the `transaction_analysis_output` directory:

- **Raw Transactions**: `credit_card_transactions.csv`  
- **Processed Transactions**: `processed_transactions.csv`  
- **Model Reports**: `{model_type}_evaluation_report.txt`  

---

## 4. Delegation of Authority Validation

The DOA policy is defined in `delegation_of_authority_policy.md` and enforced by `doa_validator.py` during preprocessing.

### How DOA Validation Works

- **Role-Based Limits**:  
  Ensures transactions do not exceed role-specific limits.

- **Approval Requirements**:  
  Validates that high-value transactions receive the required approvals.

- **Timing Rules**:  
  Checks if transactions occur within business hours (6 AM - 8 PM EST).

---

## 5. Colab Notebook Instructions

### File Requirements

- **Colab Notebook**: `DATA-5000-final-project-colab.ipynb`  
- **Dataset**: `creditcard.csv` (downloaded from Kaggle)  

### Steps to Run in Colab

1. **Upload Files to Colab**:  
   Upload `DATA-5000-final-project-colab.ipynb` and `creditcard.csv` to your Colab environment.

2. **Mount Google Drive** (optional):  
   If you want to save outputs to Google Drive, mount it with:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Load the CSV File**:  
   Ensure the path to `creditcard.csv` is correct. Update the file path in the notebook:

   ```python
   df = pd.read_csv('/content/creditcard.csv')
   ```

   If using Google Drive:

   ```python
   df = pd.read_csv('/content/drive/MyDrive/path/to/creditcard.csv')
   ```

4. **Run All Notebook Cells**:  
   Execute the cells in the notebook to:

   - Load and preprocess the dataset.  
   - Train anomaly detection models.  
   - Evaluate model performance.  
   - Generate visualizations.  

5. **Generate Model Performance Report**:  
   The notebook will generate a `model_performance_report.pdf` summarizing the model evaluation metrics.

### Output

- **PDF Report**: The generated `model_performance_report.pdf` will include precision, recall, F1 score, ROC AUC, and PR AUC for each model. The best-performing model will be highlighted.

---

## 6. Files Overview

| **File**                  | **Description**                                           |
|----------------------------|-----------------------------------------------------------|
| **2main.py**              | Main script to run the entire pipeline.                   |
| **data_preprocessing.py** | Cleans data and applies DOA validation and risk detection.|
| **doa_validator.py**      | Implements delegation of authority validation logic.      |
| **transaction_generator.py** | Generates synthetic transaction data.                   |
| **anomaly_detection.py**  | Contains anomaly detection models.                        |
| **risk_pattern_detector.py** | Detects high-risk transaction patterns.                |
| **transaction_analysis.py** | Analyzes transactions and generates visualizations.     |
| **delegation_of_authority_policy.md** | Markdown document outlining DOA policies.     |
| **DATA-5000-final-project-colab.ipynb** | Colab notebook for running the framework.   |
| **creditcard.csv**        | Credit card transaction dataset (from Kaggle).            |
| **model_performance_report.pdf** | PDF summarizing model evaluation metrics.          |

---

