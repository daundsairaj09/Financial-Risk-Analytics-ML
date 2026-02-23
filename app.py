import streamlit as st
import joblib
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Configure page settings
st.set_page_config(
    page_title="Financial Risk Prediction",
    page_icon="💸",
    layout="wide",  # Changed to wide for a better dashboard feel
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a beautiful, premium aesthetic without breaking inputs
st.markdown("""
<style>
    /* Headers */
    .main-title {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        font-size: 3rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 40px;
        font-size: 1.1rem;
    }
    
    /* Result Cards */
    .prediction-card-low {
        padding: 24px;
        border-radius: 12px;
        background: linear-gradient(to right, #dcfce7, #bbf7d0);
        color: #166534;
        text-align: center;
        border: 1px solid #86efac;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .prediction-card-high {
        padding: 24px;
        border-radius: 12px;
        background: linear-gradient(to right, #fee2e2, #fca5a5);
        color: #991b1b;
        text-align: center;
        border: 1px solid #fca5a5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .prediction-prob {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Make st.metric text slightly larger */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------
# DATA LOADING
# -----------------
@st.cache_resource
def load_models():
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    encoders = joblib.load(os.path.join(models_dir, "encoders.pkl"))
    feature_columns = joblib.load(os.path.join(models_dir, "feature_columns.pkl"))
    return rf_model, scaler, encoders, feature_columns

try:
    rf_model, scaler, encoders, feature_columns = load_models()
except FileNotFoundError:
    st.error("🚨 Model files not found! Please run the training script (`python src/train_model.py`) to generate the required model artifacts.")
    st.stop()

# -----------------
# SIDEBAR
# -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100) # Generic fast icon
    st.title("Risk Dashboard")
    st.markdown("Use this tool to evaluate the default probability of prospective loan applicants.")
    st.markdown("---")
    st.markdown("**Model Specs:**")
    st.markdown("- Algorithm: Random Forest")
    st.markdown("- Accuracy: 92.9%")
    st.markdown("- Risk Threshold: 50%")
    st.markdown("---")
    st.info("Developed for TY Project")

# -----------------
# MAIN LAYOUT
# -----------------
st.markdown('<h1 class="main-title">AI Loan Risk Assessor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter the applicant\'s details to instantly evaluate their loan default risk.</p>', unsafe_allow_html=True)

# Define Dashboard layout using Tabs
# To simulate auto-switching, we will use a session_state variable to control the default active tab if possible.
# Since Streamlit 1.0+ tabs don't allow programmatic overriding easily, we will dynamically show the result
# at the top of the page when a prediction is made.

tab1, tab2 = st.tabs(["📋 Applicant & Loan Entry", "📊 Risk Analysis & Explainability"])

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("👤 Applicant Profile")
        person_age = st.number_input("Age", min_value=18, max_value=120, value=25)
        person_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=50000000, value=650000, step=50000)
        person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5, step=1)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    with col2:
        st.subheader("🏦 Financial & Loan Request")
        loan_amnt = st.number_input("Requested Loan Amount (₹)", min_value=1000, max_value=10000000, value=150000, step=10000)
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_int_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=50.0, value=10.5, step=0.5)

    st.markdown("---")
    st.subheader("💳 Credit History")
    col3, col4 = st.columns(2)
    with col3:
        cb_person_default_on_file = st.selectbox("Prior Default on File?", ["Y", "N"])
    with col4:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50, value=4)

    # Dynamic Metrics Calculation
    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Evaluate Default Risk", type="primary")

with tab2:
    if not st.session_state.get("prediction_made"):
        st.info("👈 Please enter applicant details in the first tab and click 'Evaluate Default Risk'.")
    else:
        st.subheader("Risk Assessment Results")
        
        # Display key metrics at the top
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="Debt-to-Income Ratio", value=f"{loan_percent_income:.1%}", delta="High Risk (>40%)" if loan_percent_income > 0.4 else "Healthy")
        with metric_col2:
            st.metric(label="Requested Amount", value=f"₹{loan_amnt:,.2f}")
        with metric_col3:
            st.metric(label="Interest Rate", value=f"{loan_int_rate:.1f}%")

        st.markdown("---")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        prob = st.session_state.last_probability
        
        with res_col1:
            st.markdown("### Final Decision")
            if prob < 0.5:
                risk_level = "Loan Approved"
                css_class = "prediction-card-low"
                icon = "🎉"
            else:
                risk_level = "Loan Rejected"
                css_class = "prediction-card-high"
                icon = "🛑"
                
            st.markdown(f'''
            <div class="{css_class}">
                <div class="prediction-title">{icon} {risk_level}</div>
                <div class="prediction-prob">Estimated Default Probability: <strong>{prob:.1%}</strong></div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Actionable feedback for rejected loans
            if prob >= 0.5:
                st.markdown("<br>", unsafe_allow_html=True)
                st.error("**Why was this declined?**")
                st.markdown("Based on your profile, here is what you can improve for next time:")
                
                if loan_percent_income > 0.35:
                    st.markdown("- **High Debt Burden**: The loan amount is very high compared to your income. *Tip: Try requesting a smaller loan amount.*")
                if cb_person_default_on_file == "Y":
                    st.markdown("- **Prior Default**: Your record shows a previous failure to repay a loan. *Tip: Continue building a clean history of on-time payments.*")
                if loan_grade in ["D", "E", "F", "G"]:
                    st.markdown("- **Low Credit Grade**: Your credit profile is considered high risk. *Tip: Work on improving your overall credit score.*")
                if person_emp_length < 2:
                    st.markdown("- **Short Employment History**: You have less than 2 years of continuous employment. *Tip: A longer, stable job history increases trust.*")
                if loan_percent_income <= 0.35 and cb_person_default_on_file == "N" and loan_grade not in ["D", "E", "F", "G"] and person_emp_length >= 2:
                    st.markdown("- **Overall Risk Profile**: Your combined financial details fell just short of our safety threshold. *Tip: Increasing your income or savings can help.*")

        with res_col2:
            st.markdown("### AI Confidence Gauge")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Default Risk (%)"},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)"}, # hide default bar
                    'steps': [
                        {'range': [0, 50], 'color': "#bbf7d0"},   # Green (Safe)
                        {'range': [50, 100], 'color': "#fca5a5"}], # Red (Danger)
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob * 100
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        with st.expander("🔍 Understand This Prediction"):
            st.markdown("This chart breaks down exactly why the AI made its decision. Each bar shows the **percentage of influence** a specific piece of information had. A larger percentage means that factor mattered the most to the AI.")
            
            # Extract Feature Importances from the Random Forest model
            importances = rf_model.feature_importances_
            
            # Create a DataFrame for Plotly
            importance_df = pd.DataFrame({
                'Feature': feature_columns,
                'Importance': importances
            })
            
            # Sort by importance and pick top 8 for clarity
            importance_df = importance_df.sort_values(by='Importance', ascending=True).tail(8)
            
            # Convert importance to a percentage between 0 and 100
            importance_df['Importance %'] = importance_df['Importance'] * 100
            
            # Clean up feature names for display
            clean_names = {
                'person_income': 'Annual Income',
                'loan_percent_income': 'Debt-to-Income Ratio',
                'loan_int_rate': 'Interest Rate',
                'loan_grade': 'Loan Grade',
                'person_home_ownership': 'Home Ownership',
                'loan_amnt': 'Loan Amount',
                'cb_person_cred_hist_length': 'Credit History Length',
                'person_age': 'Age',
                'loan_intent': 'Loan Intent',
                'person_emp_length': 'Employment Length',
                'cb_person_default_on_file': 'Prior Default on File'
            }
            importance_df['Feature'] = importance_df['Feature'].map(lambda x: clean_names.get(x, x))
            
            # Build horizontal bar chart
            fig_importance = go.Figure(go.Bar(
                x=importance_df['Importance %'],
                y=importance_df['Feature'],
                orientation='h',
                text=importance_df['Importance %'].apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                marker=dict(
                    color=importance_df['Importance %'],
                    colorscale='Viridis'
                )
            ))
            
            fig_importance.update_layout(
                title='What Influenced Your Result the Most?',
                xaxis_title='Impact on Decision (%)',
                yaxis_title='',
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)

# -----------------
# PREDICTION LOGIC EVENT
# -----------------
if analyze_button:
    try:
        input_df = pd.DataFrame({
            'person_age': [person_age],
            'person_income': [person_income],
            'person_home_ownership': [person_home_ownership],
            'person_emp_length': [person_emp_length],
            'loan_intent': [loan_intent],
            'loan_grade': [loan_grade],
            'loan_amnt': [loan_amnt],
            'loan_int_rate': [loan_int_rate],
            'loan_percent_income': [loan_percent_income],
            'cb_person_default_on_file': [cb_person_default_on_file],
            'cb_person_cred_hist_length': [cb_person_cred_hist_length]
        })
        
        for col in ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']:
            input_df[col] = encoders[col].transform(input_df[col])
            
        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)
        
        prob = rf_model.predict_proba(input_scaled)[0][1]
        
        # Save to session state to show in Tab 2
        st.session_state.prediction_made = True
        st.session_state.last_probability = prob
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

