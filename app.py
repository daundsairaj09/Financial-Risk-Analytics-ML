import streamlit as st
import os
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="Enterprise Financial Risk XAI Pipeline",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #10b981, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        font-size: 3.5rem;
        font-weight: 800;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 30px;
        font-size: 1.2rem;
    }
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
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://0.0.0.0:8000")

def check_api_health():
    try:
        if requests.get(f"{API_URL}/health", timeout=2).status_code == 200:
            return True
        return False
    except:
        return False

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
    st.title("Enterprise XAI Risk Dashboard")
    st.markdown("State-of-the-Art ML Pipeline for autonomous risk assessment.")
    st.markdown("---")
    st.markdown("**Engine Specs:**")
    st.markdown("- **Core:** XGBoost / Random Forest")
    st.markdown("- **XAI:** SHAP (Game Theory)")
    st.markdown("- **GenAI:** Google Gemini Flash")
    
    if not check_api_health():
        st.error("Engine Offline. Start FastAPI.")
    else:
        st.success("API Connected: Healthy 🟢")

st.markdown('<h1 class="main-title">AI Underwriter & XAI Explainer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Next-generation transparent risk decisions powered by Explainable AI.</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📋 Application Entry", "📊 XAI XAI Analysis & Report", "🎛️ XAI What-If Simulator"])

def get_prediction(payload):
    response = requests.post(f"{API_URL}/predict", json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"API Error: {response.text}")
        return None

# Form Fields definition
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("👤 Applicant Profile")
        person_age = st.number_input("Age", 18, 120, 25)
        person_income = st.number_input("Annual Income (₹)", 10000, 50000000, 650000, step=50000)
        person_emp_length = st.number_input("Employment Length (years)", 0, 50, 5)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])

    with col2:
        st.subheader("🏦 Loan Request")
        loan_amnt = st.number_input("Requested Loan Amount (₹)", 1000, 10000000, 150000, step=10000)
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 50.0, 10.5, step=0.1)

    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        cb_person_default_on_file = st.selectbox("Prior Default on File?", ["Y", "N"])
    with col4:
        cb_person_cred_hist_length = st.number_input("Credit History Length (years)", 0, 50, 4)

    loan_percent_income = loan_amnt / person_income if person_income > 0 else 0.0

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_button = st.button("Evaluate Application XAI", type="primary", use_container_width=True)

if analyze_button:
    payload = {
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }
    st.session_state.base_payload = payload
    with st.spinner("Calculating Mathematical XAI XAI SHAP Values and querying Gemini..."):
        data = get_prediction(payload)
        if data:
            st.session_state.prediction_made = True
            st.session_state.last_probability = data.get("probability", 0.0)
            st.session_state.optimal_interest_rate = data.get("optimal_interest_rate", None)
            st.session_state.feature_importances = data.get("feature_importances", {})
            st.session_state.shap_base_value = data.get("shap_base_value", 0.5)
            st.session_state.ai_advice = data.get("ai_advice", "")
            
            # Fetch PDF
            pdf_req = requests.post(f"{API_URL}/generate_report", json={
                "applicant_data": payload,
                "probability": st.session_state.last_probability,
                "ai_advice": st.session_state.ai_advice
            })
            if pdf_req.status_code == 200:
                st.session_state.pdf_bytes = pdf_req.content

with tab2:
    if not st.session_state.get("prediction_made"):
        st.info("👈 Please evaluate an application in the first tab.")
    else:
        prob = st.session_state.last_probability
        st.subheader("Enterprise Risk Assessment")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            if prob < 0.5:
                st.markdown(f'<div class="prediction-card-low"><div class="prediction-title">✅ APPROVED</div><div class="prediction-prob">Risk Probability: {prob:.1%}</div></div>', unsafe_allow_html=True)
                opt_rate = st.session_state.get('optimal_interest_rate')
                if opt_rate:
                    st.success(f"🎯 **AI Optimization Engine:** Based on this risk profile, the mathematically optimal personalized Interest Rate is **{opt_rate}%**.")
            else:
                st.markdown(f'<div class="prediction-card-high"><div class="prediction-title">🛑 REJECTED</div><div class="prediction-prob">Risk Probability: {prob:.1%}</div></div>', unsafe_allow_html=True)
            
            st.markdown("### 🤖 Generative AI Underwriter XAI Notes")
            st.info(st.session_state.ai_advice)
            
            if "pdf_bytes" in st.session_state:
                st.download_button("📥 Download Official XAI PDF Report", st.session_state.pdf_bytes, "Risk_Report.pdf", "application/pdf", use_container_width=True)

        with c2:
            st.markdown("### XAI Mathematical Proof (SHAP Waterfall)")
            st.markdown("*Industry standard XAI game-theoretic AI explanation showing exactly how the model arrived at its probability.*")
            
            importances = st.session_state.feature_importances
            base_val = st.session_state.shap_base_value
            
            clean_names = {
                'person_income': 'Income',
                'loan_percent_income': 'DTI Ratio',
                'loan_int_rate': 'Interest Rate',
                'loan_grade': 'Loan Grade',
                'person_home_ownership': 'Home Ownership',
                'loan_amnt': 'Loan Amount',
                'cb_person_cred_hist_length': 'Credit History',
                'person_age': 'Age',
                'loan_intent': 'Loan Intent',
                'person_emp_length': 'Employment Length',
                'cb_person_default_on_file': 'Prior Default'
            }
            
            # Sort top 7 XAI SHAP features by absolute driving force
            sorted_feats = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:7]
            
            measures = ["relative"] * len(sorted_feats)
            measures.append("total")
            
            y_ax = [clean_names.get(k, k) for k, v in sorted_feats]
            y_ax.append("Final Model Value")
            
            # Waterfall x needs to be the exact SHAP contribution
            x_ax = [v for k, v in sorted_feats]
            # SHAP returns raw values, sum of base + shap = margin/prob. We assume probability for chart representation simplicity
            x_ax.append(0) # Total bar calculates itself from relative values + base
            
            fig = go.Figure(go.Waterfall(
                orientation="h",
                measure=measures,
                y=y_ax,
                x=x_ax,
                base=base_val,
                decreasing={"marker":{"color":"#10b981"}}, # Green decreases risk
                increasing={"marker":{"color":"#ef4444"}}, # Red increases risk
                totals={"marker":{"color":"#3b82f6"}}
            ))
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    if not st.session_state.get("prediction_made"):
        st.info("👈 Please evaluate an application in the first tab to unlock the What-If Simulator.")
    else:
        st.subheader("Interactive What-If XAI Simulation")
        st.markdown("Change the XAI parameters dynamically below to see how the XAI mathematical decision engine reacts. This proves the exact breaking points of the model's logic for local specific XAI explainability.")
        
        sim_payload = st.session_state.base_payload.copy()
        
        scol1, scol2 = st.columns(2)
        with scol1:
            sim_income = st.slider("Simulate XAI Income Change (₹)", 100000, 5000000, int(sim_payload['person_income']), step=50000)
            sim_payload['person_income'] = sim_income
            sim_payload['loan_percent_income'] = sim_payload['loan_amnt'] / sim_income
            st.metric("Simulated DTI Ratio", f"{sim_payload['loan_percent_income']*100:.1f}%")
            
        with scol2:
            sim_loan = st.slider("Simulate XAI Loan Request Change (₹)", 10000, 2000000, int(sim_payload['loan_amnt']), step=10000)
            sim_payload['loan_amnt'] = sim_loan
            sim_payload['loan_percent_income'] = sim_loan / sim_payload['person_income']
            
            sim_rate = st.slider("Simulate XAI Interest Rate Change (%)", 1.0, 30.0, float(sim_payload['loan_int_rate']), step=0.5)
            sim_payload['loan_int_rate'] = sim_rate
            
        if st.button("Run XAI Simulation"):
            with st.spinner("Re-evaluating mathematical SHAP landscape..."):
                sim_data = get_prediction(sim_payload)
                if sim_data:
                    prob = sim_data['probability']
                    st.markdown("### Simulated Risk Output")
                    if prob < 0.5:
                        st.success(f"✅ Approved! New Default Risk: {prob:.1%}")
                    else:
                        st.error(f"🛑 Rejected! New Default Risk: {prob:.1%}")
                        
                    # Also plot simulated XAI waterfall
                    sim_importances = sim_data['feature_importances']
                    sim_sorted = sorted(sim_importances.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    
                    s_measures = ["relative"] * 5 + ["total"]
                    s_y = [clean_names.get(k, k) for k, v in sim_sorted] + ["Final"]
                    s_x = [v for k, v in sim_sorted] + [0]
                    
                    s_fig = go.Figure(go.Waterfall(
                        orientation="h", measure=s_measures, y=s_y, x=s_x, base=sim_data['shap_base_value'],
                        decreasing={"marker":{"color":"#10b981"}}, increasing={"marker":{"color":"#ef4444"}}, totals={"marker":{"color":"#3b82f6"}}
                    ))
                    s_fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(s_fig, use_container_width=True)
