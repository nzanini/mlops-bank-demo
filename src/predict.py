# src/predict.py

import sys
import joblib
import pandas as pd

# Path to the saved model
MODEL_PATH = "models/rf_model.joblib"

def predict(input_csv: str):
    """
    Load the model and run predictions on the input CSV.
    Prints the first few rows of the DataFrame with a new 'prediction' column.
    """
    # Load input features
    df = pd.read_csv(input_csv)
    # Load trained model from disk
    model = joblib.load(MODEL_PATH)
    # Perform prediction
    preds = model.predict(df)
    # Add predictions as a new column
    df["prediction"] = preds
    # Show the top 5 rows with predictions
    print(df.head())

if __name__ == "__main__":
    # Basic CLI argument check
    if len(sys.argv) != 2:
        print("Usage: python predict.py <input_csv>")
        sys.exit(1)
    # Call predict with the provided CSV path
    predict(sys.argv[1])
