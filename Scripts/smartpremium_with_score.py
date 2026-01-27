import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# =====================
# Load data
# =====================
base_path = r"C:\Users\Lenovo\OneDrive\Desktop\SmartPremium\data\playground-series-s4e12"

train = pd.read_csv(f"{base_path}\\train.csv")
test = pd.read_csv(f"{base_path}\\test.csv")
sample = pd.read_csv(f"{base_path}\\sample_submission.csv")

# =====================
# Features & Target
# =====================
y = train["Premium Amount"]
X = train.drop(columns=["id", "Premium Amount"])
X_test = test.drop(columns=["id"])

# Drop date
X.drop(columns=["Policy Start Date"], inplace=True)
X_test.drop(columns=["Policy Start Date"], inplace=True)

# =====================
# Encode categorical safely
# =====================
cat_cols = X.select_dtypes(include="object").columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X[cat_cols] = encoder.fit_transform(X[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# =====================
# Train / Validation split
# =====================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# Model
# =====================
model = RandomForestRegressor(
    n_estimators=80,
    max_depth=14,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# Validation score
# =====================
val_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_pred))

print(f"✅ Validation RMSE: {rmse:.4f}")

# =====================
# Final train & submit
# =====================
final_pred = model.predict(X_test)
sample["Premium Amount"] = final_pred
sample.to_csv("submission.csv", index=False)

print("✅ submission.csv updated")




