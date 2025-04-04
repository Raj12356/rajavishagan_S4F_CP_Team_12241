import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve
)

# Load Dataset
file_path = "/content/solar_panel_sensor_data_processed1.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(columns=["Timestamp"], errors='ignore')

# Define features and target
X = df[["Temperature (°C)", "Voltage (V)", "Current (A)"]]
y = df["Failure"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Initial Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)

# Feature Importance Analysis
importances = rf.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(X.columns, importances, color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance for Inverter Failure Prediction")
plt.show()

# Handle Class Imbalance using Oversampling
df_majority = df[df["Failure"] == 0]
df_minority = df[df["Failure"] == 1]
df_minority_oversampled = resample(
    df_minority, replace=True, n_samples=len(df_majority), random_state=42
)
df_balanced = pd.concat([df_majority, df_minority_oversampled])

# Split balanced dataset
X_balanced = df_balanced[["Temperature (°C)", "Voltage (V)", "Current (A)"]]
y_balanced = df_balanced["Failure"]
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42
)
X_train_bal_scaled = scaler.fit_transform(X_train_bal)
X_test_bal_scaled = scaler.transform(X_test_bal)

# Retrain Model on Balanced Data
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42)
rf_balanced.fit(X_train_bal_scaled, y_train_bal)
y_pred_balanced = rf_balanced.predict(X_test_bal_scaled)
y_proba_balanced = rf_balanced.predict_proba(X_test_bal_scaled)[:, 1]

# Model Performance Evaluation
accuracy = accuracy_score(y_test_bal, y_pred_balanced)
precision = precision_score(y_test_bal, y_pred_balanced)
recall = recall_score(y_test_bal, y_pred_balanced)
f1 = f1_score(y_test_bal, y_pred_balanced)
# ... [same code as before]

# Line Graph: Probability of Failure across Test Samples
plt.figure(figsize=(10, 4))
plt.plot(y_proba_balanced, label='Predicted Probability of Failure', color='red', linewidth=1.5)
plt.scatter(range(len(y_test_bal)), y_test_bal, label='Actual Failure (0/1)', color='blue', s=10, alpha=0.6)
plt.xlabel("Test Sample Index")
plt.ylabel("Probability / Actual Label")
plt.title("Predicted Failure Probability vs Actual Label")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Q1 Answer
print("1. Most important features for predicting failures:")
print("→ Voltage and Current are the key predictors.")

# Q2 Answer
print("\n2. Model used to predict inverter failure:")
print("→ Random Forest Classifier achieved good performance.")
print(f"→ Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")

# Q3 Answer
print("\n3. Ways to improve model accuracy:")
print("→ Add more features, tune hyperparameters, or use XGBoost.")

# Q4 Answer
print("\n4. Actions to take if failure is predicted:")
print("→ Notify maintenance, check system, activate backup.")

# Q5 Answer
print("\n5. How predictive maintenance helps sustainability:")
print("→ Reduces waste, extends equipment life, saves resources.")

