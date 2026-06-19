import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
import joblib

# Create directory for plots and models if they don't exist
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------------------------------------------------------------
# 1. Load and Clean Data
# -------------------------------------------------------------
print("Loading data.csv...")
df = pd.read_csv("data.csv")

# Clean column names (strip whitespace and trailing commas)
df.columns = [col.strip().rstrip(',') for col in df.columns]

# Drop ID column if it exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

# The trailing comma in header might create an extra empty column (like 'Unnamed: 32')
unnamed_cols = [col for col in df.columns if "Unnamed" in col or col == ""]
if unnamed_cols:
    print(f"Dropping empty/unnamed columns: {unnamed_cols}")
    df = df.drop(columns=unnamed_cols)

print(f"Dataset Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("Checking for missing values:")
print(df.isnull().sum().sum(), "missing values found.")

# Encode diagnosis: M = 1, B = 0
df['diagnosis_numeric'] = df['diagnosis'].map({'M': 1, 'B': 0})

# -------------------------------------------------------------
# 2. Classification Task: Predict Diagnosis (M vs B)
# -------------------------------------------------------------
print("\n--- Running Classification Task ---")

# Define features and target
X_class = df.drop(columns=["diagnosis", "diagnosis_numeric"])
y_class = df["diagnosis_numeric"]

# Split data
X_class_train, X_class_test, y_class_train, y_class_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Scale features
scaler_class = StandardScaler()
X_class_train_scaled = scaler_class.fit_transform(X_class_train)
X_class_test_scaled = scaler_class.transform(X_class_test)

# Train models
log_reg = LogisticRegression(random_state=42, max_iter=1000)
rf_class = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)

log_reg.fit(X_class_train_scaled, y_class_train)
rf_class.fit(X_class_train_scaled, y_class_train)

# Evaluate Logistic Regression
y_pred_lr = log_reg.predict(X_class_test_scaled)
y_prob_lr = log_reg.predict_proba(X_class_test_scaled)[:, 1]

lr_metrics = {
    "Accuracy": accuracy_score(y_class_test, y_pred_lr),
    "Precision": precision_score(y_class_test, y_pred_lr),
    "Recall": recall_score(y_class_test, y_pred_lr),
    "F1-Score": f1_score(y_class_test, y_pred_lr),
    "ROC AUC": roc_auc_score(y_class_test, y_prob_lr)
}

# Evaluate Random Forest
y_pred_rf = rf_class.predict(X_class_test_scaled)
y_prob_rf = rf_class.predict_proba(X_class_test_scaled)[:, 1]

rf_metrics = {
    "Accuracy": accuracy_score(y_class_test, y_pred_rf),
    "Precision": precision_score(y_class_test, y_pred_rf),
    "Recall": recall_score(y_class_test, y_pred_rf),
    "F1-Score": f1_score(y_class_test, y_pred_rf),
    "ROC AUC": roc_auc_score(y_class_test, y_prob_rf)
}

print("\nLogistic Regression Metrics:")
for k, v in lr_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nRandom Forest Classifier Metrics:")
for k, v in rf_metrics.items():
    print(f"  {k}: {v:.4f}")

# Choose best classifier based on F1-Score
best_class_model = log_reg if lr_metrics["F1-Score"] >= rf_metrics["F1-Score"] else rf_class
best_class_name = "Logistic Regression" if best_class_model == log_reg else "Random Forest"
print(f"\nBest Classification Model chosen: {best_class_name}")

# Save classification assets
joblib.dump(best_class_model, "models/classifier.joblib")
joblib.dump(scaler_class, "models/scaler_class.joblib")

# Save Confusion Matrix plot
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_class_test, best_class_model.predict(X_class_test_scaled))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.title(f"Confusion Matrix - {best_class_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

# Save Feature Importance plot for Random Forest
plt.figure(figsize=(10, 6))
importances = rf_class.feature_importances_
indices = np.argsort(importances)[::-1]
sns.barplot(x=importances[indices][:10], y=X_class.columns[indices][:10], palette="viridis")
plt.title("Top 10 Feature Importances (Random Forest Classifier)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("plots/classification_feature_importance.png", dpi=150)
plt.close()


# -------------------------------------------------------------
# 3. Regression Task: Predict Radius Mean
# -------------------------------------------------------------
print("\n--- Running Regression Task ---")

# Define target and features for predicting radius_mean
# Target: radius_mean
# Features: shape features + diagnosis. Avoid area/perimeter/worst/se to make it a non-trivial prediction
reg_features = [
    "texture_mean", "smoothness_mean", "compactness_mean", "concavity_mean", 
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "diagnosis_numeric"
]
X_reg = df[reg_features]
y_reg = df["radius_mean"]

# Split data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# Train models
ridge_reg = Ridge(alpha=1.0, random_state=42)
rf_reg = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5)

ridge_reg.fit(X_reg_train_scaled, y_reg_train)
rf_reg.fit(X_reg_train_scaled, y_reg_train)

# Evaluate Ridge Regression
y_pred_ridge = ridge_reg.predict(X_reg_test_scaled)
ridge_metrics = {
    "MAE": mean_absolute_error(y_reg_test, y_pred_ridge),
    "MSE": mean_squared_error(y_reg_test, y_pred_ridge),
    "RMSE": np.sqrt(mean_squared_error(y_reg_test, y_pred_ridge)),
    "R2": r2_score(y_reg_test, y_pred_ridge)
}

# Evaluate Random Forest Regressor
y_pred_rf_reg = rf_reg.predict(X_reg_test_scaled)
rf_reg_metrics = {
    "MAE": mean_absolute_error(y_reg_test, y_pred_rf_reg),
    "MSE": mean_squared_error(y_reg_test, y_pred_rf_reg),
    "RMSE": np.sqrt(mean_squared_error(y_reg_test, y_pred_rf_reg)),
    "R2": r2_score(y_reg_test, y_pred_rf_reg)
}

print("\nRidge Regression Metrics:")
for k, v in ridge_metrics.items():
    print(f"  {k}: {v:.4f}")

print("\nRandom Forest Regressor Metrics:")
for k, v in rf_reg_metrics.items():
    print(f"  {k}: {v:.4f}")

# Choose best regressor based on R2 score
best_reg_model = ridge_reg if ridge_metrics["R2"] >= rf_reg_metrics["R2"] else rf_reg
best_reg_name = "Ridge Regression" if best_reg_model == ridge_reg else "Random Forest Regressor"
print(f"\nBest Regression Model chosen: {best_reg_name}")

# Save regression assets
joblib.dump(best_reg_model, "models/regressor.joblib")
joblib.dump(scaler_reg, "models/scaler_reg.joblib")

# Save Actual vs Predicted plot
plt.figure(figsize=(6, 5))
y_pred_best_reg = best_reg_model.predict(X_reg_test_scaled)
sns.scatterplot(x=y_reg_test, y=y_pred_best_reg, alpha=0.7, color='purple')
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], '--', color='red')
plt.title(f"Actual vs Predicted - {best_reg_name}")
plt.xlabel("Actual radius_mean")
plt.ylabel("Predicted radius_mean")
plt.tight_layout()
plt.savefig("plots/regression_actual_vs_predicted.png", dpi=150)
plt.close()

# Save Feature Importance plot for Random Forest Regressor
plt.figure(figsize=(10, 6))
importances_reg = rf_reg.feature_importances_
indices_reg = np.argsort(importances_reg)[::-1]
sns.barplot(x=importances_reg[indices_reg], y=np.array(reg_features)[indices_reg], palette="plasma")
plt.title("Feature Importances (Random Forest Regressor)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig("plots/regression_feature_importance.png", dpi=150)
plt.close()

print("\nTraining completed successfully! Models and plots saved.")
