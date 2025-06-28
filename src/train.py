# src/train.py

import os
import joblib                # for saving/loading models
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow                # MLflow tracking API

# Directories
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def load_data():
    """
    Load training data from CSV and split into features X and target y.
    """
    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    X = train_df.drop(columns=["y"])  # drop the target column
    y = train_df["y"]                 # target series
    return X, y

if __name__ == "__main__":
    # Start an MLflow run context
    with mlflow.start_run():
        # Load features and labels
        X_train, y_train = load_data()
        
        # Define model hyperparameters
        n_estimators = 100   # number of trees in the forest
        max_depth = 5        # maximum depth of each tree
        
        # Log hyperparameters to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Initialize and train the RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42       # seed for reproducibility
        )
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, preds)
        # Log metric to MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        
        # Save the trained model to disk
        model_path = os.path.join(MODEL_DIR, "rf_model.joblib")
        joblib.dump(model, model_path)
        # Log the model artifact in MLflow under 'models/' folder
        mlflow.log_artifact(model_path, artifact_path="models")
        
        print(f"Training complete. Training accuracy: {train_acc:.4f}")
