# Regression on Tabular Data

## Overview
This project involves building a regression model to predict a target variable based on 53 anonymized features. The main objective is to minimize the Root Mean Squared Error (RMSE) on the test dataset.

## Repository Structure
```plaintext
regression-tabular-data/
├── data/
│   ├── train.csv
│   ├── hidden_test.csv
│   └── sample_submission.csv
├── notebooks/
│   └── EDA.ipynb
├── scripts/
│   ├── train.py
│   └── predict.py
├── predictions/
│   └── predictions.csv
├── README.md
└── requirements.txt
```

## Setup Instructions

1. Create a Virtual Environment
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Linux/macOS
   venv\Scripts\activate      # For Windows
   ```

2. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Notebook for EDA
   Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

4. Train the Model
   Run the training script:
   ```bash
   python scripts/train.py --train_path data/train.csv --model_path models/random_forest.joblib
   ```

5. Generate Predictions
   Run the prediction script:
   ```bash
   python scripts/predict.py --test_path data/hidden_test.csv --model_path models/random_forest.joblib --output_path predictions/predictions.csv
   ```

## Requirements
The dependencies are listed in the `requirements.txt` file:
```plaintext
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
joblib==1.3.0
jupyter==1.0.0
```

