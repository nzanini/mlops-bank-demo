# src/preprocessing.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths to raw and processed data
RAW_CSV = "data/raw/bank.csv"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Encode target column 'y' to numeric (yes→1, no→0)
    - One-hot encode all object (categorical) columns, dropping the first
      dummy to avoid the dummy-variable trap.
    """
    # Map yes/no to 1/0
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    
    # Identify all categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    # pd.get_dummies does one-hot encoding
    # drop_first=True drops one category per feature to prevent multicollinearity
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

if __name__ == "__main__":
    # Load raw CSV into DataFrame
    df = pd.read_csv(RAW_CSV)
    # Apply preprocessing
    df_processed = preprocess(df)
    # Split into train (80%) and test (20%)
    train_df, test_df = train_test_split(
        df_processed, test_size=0.2, random_state=42
    )
    # Save to disk
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    print("Preprocessing complete. Train/test CSVs saved.")
