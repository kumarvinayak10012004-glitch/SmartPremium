import pandas as pd
import pickle
import os

MODEL_PATH = "artifacts/model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


def predict(test_csv_path, output_path="submission.csv"):
    model = load_model()

    df_test = pd.read_csv(test_csv_path)

    # ID column (Kaggle pattern)
    test_ids = df_test.iloc[:, 0]
    X_test = df_test.drop(columns=[df_test.columns[0]])

    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "id": test_ids,
        "target": predictions
    })

    submission.to_csv(output_path, index=False)
    print("✅ submission.csv created successfully")
