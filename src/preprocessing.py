import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# 🔹 TRAINING TIME
def preprocess_train(df, target_col):

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    selected_cols = [
        "Age",
        "Gender",
        "Annual Income",
        "Health Score",
        "Smoking Status",
        "Exercise Frequency",
        "Previous Claims",
        "Vehicle Age",
        "Credit Score",
        "Insurance Duration",
        "Policy Type",
        target_col
    ]

    df = df[selected_cols]

    y = df[target_col].values
    X_df = df.drop(columns=[target_col])

    cat_cols = X_df.select_dtypes(include="object").columns.tolist()
    num_cols = X_df.select_dtypes(exclude="object").columns.tolist()

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num = num_imputer.fit_transform(X_df[num_cols])
    X_cat_raw = cat_imputer.fit_transform(X_df[cat_cols])

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    X_cat = encoder.fit_transform(X_cat_raw)

    X = np.hstack([X_num, X_cat])

    return X, y, encoder, num_imputer, cat_imputer


# 🔹 PREDICTION TIME
def preprocess_predict(df, encoder, num_imputer, cat_imputer):

    X_df = df.copy()

    cat_cols = X_df.select_dtypes(include="object").columns.tolist()
    num_cols = X_df.select_dtypes(exclude="object").columns.tolist()

    X_num = num_imputer.transform(X_df[num_cols])
    X_cat_raw = cat_imputer.transform(X_df[cat_cols])
    X_cat = encoder.transform(X_cat_raw)

    X = np.hstack([X_num, X_cat])

    return X







