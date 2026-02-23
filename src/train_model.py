import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading data...")
    # Go up one directory to access data/raw (assuming this is run from the project root)
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "credit_risk_dataset.csv")
    
    if not os.path.exists(data_path):
        # Fallback if run directly from src/
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "credit_risk_dataset.csv")
        
    df = pd.read_csv(data_path)

    print("Handling missing values...")
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    print("Encoding categorical variables...")
    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    print("Splitting dataset...")
    X = df.drop('loan_status', axis=1)
    y_classification = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_classification, test_size=0.2, random_state=42
    )

    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training RandomForest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    accuracy = rf_model.score(X_test_scaled, y_test)
    print(f"Model trained successfully! Test Accuracy: {accuracy:.4f}")

    # Save artifacts
    print("Saving model and preprocessors to 'models/' directory...")
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(models_dir, "encoders.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, "feature_columns.pkl")) # Save feature order
    
    print("All artifacts saved successfully!")

if __name__ == "__main__":
    main()
