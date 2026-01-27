import pandas as pd
import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.preprocessing import preprocess_train

# -----------------------------
# 1️⃣ Load data
# -----------------------------
data_path = "data/playground-series-s4e12/train.csv"
df = pd.read_csv(data_path)

# -----------------------------
# 2️⃣ Preprocess (TRAINING)
# -----------------------------
X, y, encoder, num_imputer, cat_imputer = preprocess_train(
    df,
    target_col="Premium Amount"
)

# -----------------------------
# 3️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4️⃣ Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 5️⃣ Evaluate
# -----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ RMSE: {rmse}")

# -----------------------------
# 6️⃣ Save artifacts
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "artifacts/model.pkl")
joblib.dump(encoder, "artifacts/encoder.pkl")
joblib.dump(num_imputer, "artifacts/num_imputer.pkl")
joblib.dump(cat_imputer, "artifacts/cat_imputer.pkl")

print("✅ Model, encoder & imputers saved successfully")






