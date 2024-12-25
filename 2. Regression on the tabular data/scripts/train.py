import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Train a regression model.')
    parser.add_argument('--train_path', type=str, default='../regression-tabular-data/data/train.csv', help='Path to training data.')
    parser.add_argument('--model_path', type=str, default='../models/random_forest.joblib', help='Path to save the trained model.')
    return parser.parse_args()

def main():
    args = parse_args()
    
    train_path = os.path.abspath(args.train_path)
    model_path = os.path.abspath(args.model_path)
    
    print(f"Loading training data from: {train_path}")

    if not os.path.isfile(train_path):
        print(f"Error: The training file '{train_path}' does not exist.")
        sys.exit(1)
    
    try:
        data = pd.read_csv(train_path)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error reading the training file: {e}")
        sys.exit(1)
    
    if 'target' not in data.columns:
        print("Error: 'target' column not found in the training data.")
        sys.exit(1)

    X = data.drop('target', axis=1)
    y = data['target']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")
    
    predictions = model.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    rmse = np.sqrt(mse)
    print(f'Validation RMSE: {rmse}')

    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created model directory: {model_dir}")

    joblib.dump(model, model_path)
    print(f'Model saved to {model_path}')

if __name__ == '__main__':
    main()
