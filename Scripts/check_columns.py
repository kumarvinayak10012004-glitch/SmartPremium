import pandas as pd

# Base folder jahan files hain
base_path = r"C:\Users\Lenovo\OneDrive\Desktop\SmartPremium\data\playground-series-s4e12"

# Load files
train = pd.read_csv(f"{base_path}\\train.csv")
test = pd.read_csv(f"{base_path}\\test.csv")
sample = pd.read_csv(f"{base_path}\\sample_submission.csv")

# Check columns
print("Train columns:", train.columns)
print("Test columns:", test.columns)
print("Sample Submission columns:", sample.columns)
