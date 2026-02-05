# 💰 SmartPremium: Predicting Insurance Costs with Machine Learning

## 📌 Project Overview
SmartPremium is a machine learning project designed to predict insurance premium costs based on customer demographics, financial status, health indicators, and policy details. The goal is to build a data-driven system that helps insurance companies estimate premiums accurately and efficiently.

This project demonstrates the complete machine learning lifecycle including data preprocessing, exploratory data analysis, model development, experiment tracking using MLflow, and deployment through a Streamlit web application.

---

## 🎯 Problem Statement
Insurance companies rely on various risk factors such as age, income, health status, and claim history to determine premium costs. The objective of this project is to develop a robust regression model capable of predicting insurance premiums based on real-world customer and policy data.

---

## 🏢 Business Use Cases
- 💰 Insurance Companies: Optimize premium pricing based on customer risk profiles
- 📊 Financial Institutions: Assist in risk assessment for insurance-linked financial products
- 🧑‍⚕️ Healthcare Providers: Estimate future healthcare-related insurance costs
- 🔍 Customer Service Platforms: Provide real-time insurance quotes

---

## 🧠 Skills Demonstrated
- Data Preprocessing & Feature Engineering
- Exploratory Data Analysis (EDA)
- Regression Modeling & Evaluation
- Hyperparameter Tuning
- ML Pipelines & MLflow Integration
- Streamlit Web App Deployment
- Version Control using Git & GitHub

---

## 🏗️ Project Workflow

### 📌 Step 1: Data Understanding & EDA
- Dataset exploration and structure analysis
- Missing value detection
- Feature distribution analysis
- Correlation analysis
- Data visualization for insights

### 📌 Step 2: Data Preprocessing
- Missing value handling (Median / Mode Imputation)
- Categorical feature encoding
- Feature scaling
- Train-test split (80/20)
- Data cleaning & transformation

### 📌 Step 3: Model Development
Regression models used:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

Evaluation Metrics:
- RMSE
- RMSLE
- MAE
- R² Score

### 📌 Step 4: ML Pipeline & MLflow
- Automated ML pipeline creation
- Experiment tracking with MLflow
- Model versioning and logging
- Performance comparison

### 📌 Step 5: Deployment
- Streamlit-based web application
- Real-time insurance premium prediction
- User-friendly input interface

---

## 📊 Results
- Achieved low prediction error rates
- Developed a fully functional Streamlit web application
- Automated ML workflow with experiment tracking

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-Learn
- XGBoost
- MLflow
- Streamlit
- Git & GitHub

---

## 📂 Dataset Information
- Format: CSV
- Records: 200K+ entries
- Features: 20+ mixed data types
- Target Variable: Premium Amount

### Key Features
- Age
- Gender
- Annual Income
- Marital Status
- Education Level
- Occupation
- Health Score
- Location
- Policy Type
- Previous Claims
- Credit Score
- Smoking Status
- Exercise Frequency
- Property Type
- Customer Feedback
- Policy Start Date

### Data Characteristics
- Missing Values
- Incorrect Data Types
- Skewed Distributions
- Outliers

---

## 🚀 Project Deliverables
- Jupyter Notebook with full analysis
- ML Pipeline with MLflow integration
- Trained Machine Learning Model
- Streamlit Web Application

---

## 📦 Installation & Setup

```bash
git clone https://github.com/kumarvinayak10012004-glitch/SmartPremium.git
cd SmartPremium
pip install -r requirements.txt
Run Streamlit App:

streamlit run app.py
📌 Project Structure
SmartPremium/
│
├── data/
├── notebooks/
├── models/
├── artifacts/
├── src/
├── app.py
├── requirements.txt
└── README.md


