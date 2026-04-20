import os
from google import genai
from google.genai import types

def get_financial_advice(applicant_data: dict, risk_probability: float, feature_importances: dict) -> str:
    """
    Sends the applicant's profile and the Random Forest model's findings to Gemini
    to generate an expert, personalized financial breakdown.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "⚠️ **Agent Offline:** Gemini API key not found in environment. Please set GEMINI_API_KEY to enable the AI Underwriter Assistant."

    target_model = "gemini-2.5-flash"

    try:
        # Initialize the GenAI client
        client = genai.Client(api_key=api_key)
        
        prompt = _build_underwriter_prompt(applicant_data, risk_probability, feature_importances)
        
        # Configure the LLM for a structured, professional tone
        config = types.GenerateContentConfig(
            temperature=0.3, # Low temperature for more deterministic, professional financial advice
            system_instruction=(
                "You are an expert, empathetic Senior Financial Underwriter and Credit Counselor for a modern fintech company. "
                "Your job is to read the output of our internal Random Forest risk model, analyze the applicant's data, "
                "and provide a highly personalized, easy-to-understand explanation of why they received this risk score. "
                "Always be respectful. Use formatting (bullet points, bold text) to make it highly readable. "
                "Do not hallucinate fake numbers. Base your advice STRICTLY on the provided data."
            )
        )

        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )
        
        return response.text
        
    except Exception as e:
        error_str = str(e)
        # Catch highly-demanded API limits and return a clean fallback instead of a raw JSON error
        if "503" in error_str or "UNAVAILABLE" in error_str or "429" in error_str:
            decision = "APPROVED" if risk_probability < 0.5 else "REJECTED (High Risk)"
            fallback = f"*(Note: The AI Underwriter is currently experiencing high server demand. Here is a standard automated breakdown.)*\n\n"
            fallback += f"**Risk Decision:** {decision}\n\n"
            fallback += f"This decision reflects an estimated **{risk_probability * 100:.1f}% probability of default**. "
            fallback += "The model primarily analyzed your Debt-to-Income ratio, current income, and loan grade to formulate this risk profile. "
            
            if risk_probability >= 0.5:
                fallback += "\n\n**Recommendation:** To improve your approval odds in the future, consider lowering your requested loan amount or increasing your steady income to reduce your Debt-to-Income ratio."
            else:
                fallback += "\n\n**Recommendation:** Your strong financial profile indicates reliable creditworthiness. Setting up automatic payments is recommended to maintain this good standing."
                
            return fallback
            
        return f"⚠️ **Agent Error:** Failed to generate AI advice. Details: {error_str}"

def _build_underwriter_prompt(app_data: dict, prob: float, importances: dict) -> str:
    """Helper formatting function to construct the LLM prompt"""
    is_approved = prob < 0.5
    status = "APPROVED" if is_approved else "REJECTED (High Risk)"
    
    # Sort importances to show the LLM what drove the model the most
    sorted_factors = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    top_factors_str = "\n".join([f"- {k}: {v*100:.1f}% impact" for k, v in sorted_factors])
    
    prompt = f"""
    Applicant Financial Profile:
    - Age: {app_data.get('person_age')}
    - Annual Income: ₹{app_data.get('person_income'):,.2f}
    - Employment Length: {app_data.get('person_emp_length')} years
    - Home Ownership: {app_data.get('person_home_ownership')}
    - Historical Default on File: {app_data.get('cb_person_default_on_file')}
    - Credit History Length: {app_data.get('cb_person_cred_hist_length')} years
    
    Loan Request Details:
    - Amount: ₹{app_data.get('loan_amnt'):,.2f}
    - Intent: {app_data.get('loan_intent')}
    - Grade: {app_data.get('loan_grade')}
    - Interest Rate: {app_data.get('loan_int_rate')}%
    - Debt-to-Income Ratio: {app_data.get('loan_percent_income')*100:.1f}%

    ---
    Machine Learning Model Output:
    - Default Probability: {prob * 100:.1f}%
    - Final Decision: {status}
    
    Top 5 Data Points Driving the AI's Decision:
    {top_factors_str}
    
    Task:
    Write a 3-4 paragraph summary addressed directly to the applicant ('You'). 
    1) Explain their result simply based on the probability.
    2) Break down the *why*. Specifically reference their exact data numbers (like their DTI ratio or income) and cross-reference them with the 'Top Data Points Driving the AI' to explain how the ML model arrived at its conclusion. Let the user know the ML model logic contextually.
    3) If Rejected, provide 2 actionable steps they can take over the next 6 months to improve. If Approved, provide a brief tip on maintaining their good standing.
    """
    return prompt
