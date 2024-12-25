import argparse
import pandas as pd
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions for test data.')
    parser.add_argument('--test_path', type=str, default='../regression-tabular-data/data/hidden_test.csv', help='Path to test data.')
    parser.add_argument('--model_path', type=str, default='../regression-tabular-data/models/random_forest.joblib', help='Path to the trained model.')
    parser.add_argument('--output_path', type=str, default='../regression-tabular-data/predictions/predictions.csv', help='Path to save predictions.')
    return parser.parse_args()

def main():
    args = parse_args()

    model = joblib.load(args.model_path)

    test = pd.read_csv(args.test_path)
    
    predictions = model.predict(test)

    submission = pd.DataFrame({'Id': test.index, 'Predicted': predictions})
    submission.to_csv(args.output_path, index=False)
    print(f'Predictions saved to {args.output_path}')

if __name__ == '__main__':
    main()
