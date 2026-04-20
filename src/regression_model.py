import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_regression_model():
    print("Loading raw data...")
    # Load dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "credit_risk_dataset.csv")
    df = pd.read_csv(data_path)

    # 1. Clean Data (same as classification preprocessing)
    # Drop rows with missing values
    df = df.dropna()

    # We want to predict loan_int_rate based on applicant features.
    target = 'loan_int_rate'
    
    # We remove the actual default status because in the real world, at the time of assigning an interest rate,
    # we don't know if they will default yet. We only know their prior defaults.
    features_to_drop = [target, 'loan_status'] # loan_status is the default target
    
    if target not in df.columns:
        print(f"Target '{target}' not found. Cannot train regression model.")
        return

    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    y = df[target]

    # Handle Categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Save encoders specifically for regression or just reuse the global ones if they match.
    # To be safe, we save regression-specific encoders
    feature_columns = X.columns.tolist()

    print("Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Regressor for Dynamic Interest Rates...")
    # Use fewer estimators to train quickly for the sandbox
    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Validate
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Regression Model Metrics: MAE = {mae:.2f}%, R^2 = {r2:.3f}")

    # Save artifacts
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(models_dir, "regression_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "reg_scaler.pkl"))
    joblib.dump(encoders, os.path.join(models_dir, "reg_encoders.pkl"))
    joblib.dump(feature_columns, os.path.join(models_dir, "reg_feature_columns.pkl"))
    
    print("Optimization Regression AI Model saved successfully to models/ directory.")

if __name__ == "__main__":
    train_regression_model()
