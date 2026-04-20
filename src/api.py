import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.llm_agent import get_financial_advice

load_dotenv() # Load GEMINI_API_KEY from .env locally

app = FastAPI(title="Financial Risk API", version="1.0.0", description="API for predicting loan default risk")

# Define Data Model based on the features used in model training
class ApplicantData(BaseModel):
    person_age: int = Field(..., ge=18, le=120, description="Age of the applicant")
    person_income: int = Field(..., ge=0, description="Annual income of the applicant")
    person_home_ownership: str = Field(..., description="Home ownership status (RENT, OWN, MORTGAGE, OTHER)")
    person_emp_length: float = Field(..., ge=0, description="Employment length in years")
    loan_intent: str = Field(..., description="Intent of the loan")
    loan_grade: str = Field(..., description="Grade of the loan (A-G)")
    loan_amnt: int = Field(..., gt=0, description="Loan amount requested")
    loan_int_rate: float = Field(..., ge=0, description="Interest rate of the loan")
    loan_percent_income: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    cb_person_default_on_file: str = Field(..., description="Historical default on file (Y/N)")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Credit history length in years")

# Global variables for classification
rf_model = None
scaler = None
encoders = None
feature_columns = None

# Global variables for regression
reg_model = None
reg_scaler = None
reg_encoders = None
reg_feature_columns = None

@app.on_event("startup")
def load_models():
    global rf_model, scaler, encoders, feature_columns
    global reg_model, reg_scaler, reg_encoders, reg_feature_columns
    try:
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        # Load Classification Models
        rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
        scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        encoders = joblib.load(os.path.join(models_dir, "encoders.pkl"))
        feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
        
        # Load Regression Models
        reg_model = joblib.load(os.path.join(models_dir, "regression_model.pkl"))
        reg_scaler = joblib.load(os.path.join(models_dir, "reg_scaler.pkl"))
        reg_encoders = joblib.load(os.path.join(models_dir, "reg_encoders.pkl"))
        reg_feature_columns = joblib.load(os.path.join(models_dir, "reg_feature_columns.pkl"))
        
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint to verify API and models are loaded"""
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
    return {"status": "healthy", "model": "RandomForestClassifier"}

@app.post("/predict")
def predict_risk(data: ApplicantData):
    """Endpoint to predict loan default probability"""
    if rf_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded properly")
        
    try:
        # Convert Pydantic model to Dictionary, then to DataFrame
        input_dict = data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Apply Label Encoders
        for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
            if col in encoders:
                input_df[col] = encoders[col].transform(input_df[col])
                
        # Enforce column order
        input_df = input_df[feature_columns]
        
        # Scale numerical features
        input_scaled = scaler.transform(input_df)
        
        # Predict probability of class 1 (Default)
        prob = float(rf_model.predict_proba(input_scaled)[0][1])
        
        # --- Cascading Regression Logic for Optimal Interest Rate ---
        optimal_rate = None
        if prob < 0.5: # If Approved, assign dynamic rate
            reg_dict = input_dict.copy()
            # Regression doesn't use loan_int_rate because it predicts it
            if 'loan_int_rate' in reg_dict:
                del reg_dict['loan_int_rate']
                
            reg_df = pd.DataFrame([reg_dict])
            # Apply Reg-specific Encoders
            for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
                if col in reg_encoders:
                    reg_df[col] = reg_encoders[col].transform(reg_df[col])
                    
            reg_df = reg_df[reg_feature_columns]
            reg_scaled = reg_scaler.transform(reg_df)
            optimal_rate = float(reg_model.predict(reg_scaled)[0])
            
            # Bound realistic rate
            optimal_rate = round(max(5.0, min(optimal_rate, 24.99)), 2)
        # -------------------------------------------------------------
        
        # Advanced Explainable AI (SHAP)
        import shap
        try:
            explainer = shap.TreeExplainer(rf_model)
            shap_values = explainer.shap_values(input_scaled)
            
            # Scikit-Learn RF returns a list of arrays for classification (one for each class)
            if isinstance(shap_values, list):
                shap_vals_class1 = shap_values[1][0].tolist()
                base_val = float(explainer.expected_value[1])
            elif len(shap_values.shape) == 3:
                shap_vals_class1 = shap_values[0, :, 1].tolist()
                base_val = float(explainer.expected_value[1])
            else:
                shap_vals_class1 = shap_values[0].tolist()
                base_val = float(explainer.expected_value)
                
            shap_dict = dict(zip(feature_columns, shap_vals_class1))
        except Exception as e:
            # Fallback to standard feature importances if SHAP errors out due to versioning
            print(f"SHAP XAI Warning: {e}")
            importances = rf_model.feature_importances_.tolist()
            shap_dict = dict(zip(feature_columns, importances))
            base_val = 0.5
        
        # Trigger Generative AI Agent
        ai_advice = get_financial_advice(
            applicant_data=input_dict, 
            risk_probability=prob, 
            feature_importances=shap_dict # Sending precise SHAP logic instead of generic importances
        )
        
        return {
            "probability": prob,
            "optimal_interest_rate": optimal_rate,
            "feature_importances": shap_dict, # Using precise SHAP values natively!
            "shap_base_value": base_val,
            "ai_advice": ai_advice
        }

        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

from fastapi.responses import Response
from src.pdf_generator import generate_pdf_report

class ReportRequest(BaseModel):
    applicant_data: dict
    probability: float
    ai_advice: str

@app.post("/generate_report")
def create_report(req: ReportRequest):
    """Generates a PDF report for the risk assessment"""
    try:
        pdf_bytes = generate_pdf_report(
            applicant_data=req.applicant_data,
            probability=req.probability,
            ai_advice=req.ai_advice
        )
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=Risk_Assessment_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF: {str(e)}")
