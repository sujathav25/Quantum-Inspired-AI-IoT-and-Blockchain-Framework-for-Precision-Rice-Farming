import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, cohen_kappa_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
)
# Load the dataset
data_path = r"C:\Users\pasup\Downloads\archive (3).zip2\ICRISAT-District Level Data (1).csv"

df = pd.read_csv(data_path)
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print("\nColumns:", df.columns.tolist())

# Preprocessing
# Drop missing values 
df = df.dropna()

# Encode string columns to numbers
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Detect target column automatically
possible_targets = [col for col in df.columns if 'yield' in col.lower() or 'production' in col.lower()]
if len(possible_targets) == 0:
    target_col = df.columns[-1]
else:
    target_col = possible_targets[0]

print(f"\n🎯 Target column detected: {target_col}")

X = df.drop(target_col, axis=1)
y = df[target_col]

# Detect if it's classification or regression
if y.nunique() <= 10:
    task = 'classification'
else:
    task = 'regression'
print(f"Detected Task Type: {task.upper()}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
if task == 'classification':
    model = RandomForestClassifier(random_state=42)
else:
    model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics (with FIXED MAPE)
if task == 'classification':
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "Cohen’s Kappa": cohen_kappa_score(y_test, y_pred)
    }
    try:
        y_proba = model.predict_proba(X_test)
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_proba, multi_class='ovr')
        metrics["Log Loss"] = log_loss(y_test, y_proba)
    except Exception:
        metrics["ROC-AUC"] = "N/A"
        metrics["Log Loss"] = "N/A"

else:
    # Regression metrics
    mse = mean_squared_error(y_test, y_pred)
    # Safe MAPE: Replace zeros with small epsilon to prevent division by zero
    epsilon = 1e-8
    y_test_safe = np.where(y_test == 0, epsilon, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / y_test_safe)) * 100

    metrics = {
        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred),
        "Mean Squared Error (MSE)": mse,
        "Root Mean Squared Error (RMSE)": np.sqrt(mse),
        "R² Score": r2_score(y_test, y_pred),
        "Explained Variance": explained_variance_score(y_test, y_pred),
        "Mean Absolute Percentage Error (MAPE)": mape
    }

# Display metrics
print("\n Final Evaluation Metrics:\n")
results = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
print(results.to_string(index=False))

# Save results to CSV
results.to_csv("evaluation_results.csv", index=False)
print("\n Metrics saved to 'evaluation_results.csv' in current folder.")
