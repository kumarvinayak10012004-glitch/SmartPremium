import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder

# ===============================
# STEP 1: Load data
# ===============================
base_path = r"C:\Users\Lenovo\OneDrive\Desktop\SmartPremium\data\playground-series-s4e12"

train = pd.read_csv(f"{base_path}\\train.csv")
test = pd.read_csv(f"{base_path}\\test.csv")
sample = pd.read_csv(f"{base_path}\\sample_submission.csv")

print("✅ Files loaded")

# ===============================
# STEP 2: Target & Features
# ===============================
y_train = train["Premium Amount"]

X_train = train.drop(columns=["id", "Premium Amount"])
X_test = test.drop(columns=["id"])

# ===============================
# STEP 3: Drop Date column (VERY IMPORTANT)
# ===============================
X_train.drop(columns=["Policy Start Date"], inplace=True)
X_test.drop(columns=["Policy Start Date"], inplace=True)

# ===============================
# STEP 4: Handle CATEGORICAL columns safely
# (NO LabelEncoder, NO get_dummies)
# ===============================
cat_cols = X_train.select_dtypes(include="object").columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test[cat_cols] = encoder.transform(X_test[cat_cols])

# ===============================
# STEP 5: Train model (memory safe)
# ===============================
model = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Model trained")

# ===============================
# STEP 6: Predict & save submission
# ===============================
predictions = model.predict(X_test)

sample["Premium Amount"] = predictions
sample.to_csv("submission.csv", index=False)

print("✅ submission.csv CREATED SUCCESSFULLY")
print(sample.head())



