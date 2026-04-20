# Financial Risk Analytics
## Viva / Oral Exam Preparation Guide

This document contains highly probable Viva questions related to the core concepts backing this specific project, ranging from MLOps and Architecture to pure Machine Learning theory.

---

### Category 1: MLOps, CI/CD, & Version Control

**Q1: How does your project integrate MLOps principles?**
**Answer:** MLOps (Machine Learning Operations) is about reliable deployment and maintenance. In this project, we separated the ML inference engine (FastAPI backend) from the user interface (Streamlit frontend). Crucially, we packaged both in **Docker containers**. This ensures that the environment (Python version, library dependencies) is identical in development, testing, and production phases, completely eliminating the "it works on my machine" problem.

**Q2: How would you implement a CI/CD pipeline for this application in an enterprise setting?**
**Answer:** In a real-world scenario, I would use **GitHub Actions**. 
*   **Continuous Integration (CI):** Every time code is pushed, the pipeline would automatically trigger `pytest` to run unit tests on the FastAPI endpoints to ensure the math evaluates correctly before deployment. 
*   **Continuous Deployment (CD):** Once tests pass, the pipeline would automatically build new Docker images from `Dockerfile.api` and `Dockerfile.frontend` and push them to Docker Hub or AWS ECR, and orchestrate the deployment natively.

**Q3: How do you handle Version Control for ML models, not just code?**
**Answer:** While we use **Git** to version control the `.py` scripts and UI code, ML models (the `.pkl` files) are binary blobs and can get massive. In an enterprise system, I would use a tool like **DVC (Data Version Control)** or **MLflow** to catalog the models. This would track exactly which version of the dataset trained which version of the Random Forest model.

---

### Category 2: Core ML & Model Architecture

**Q4: Why did you choose Random Forest over a standard Decision Tree or Logistic Regression?**
**Answer:** A single Decision Tree is highly prone to **overfitting**—it memorizes the training data. A Random Forest solves this by using an ensemble of many uncorrelated trees (bagging) and taking the majority vote, making it highly robust. We didn't use Logistic Regression because financial data often has **non-linear relationships** (e.g., age and risk don't scale in a straight line), which Random Forests handle natively without needing massive feature engineering.

**Q5: Explain your "Cascading Architecture". Why use both Classification and Regression?**
**Answer:** Real banking requires two steps: *Decisioning* and *Pricing*. 
1.  First, the **Classification model** (Random Forest) acts as a rigorous gatekeeper, processing the data to output a strict approve/reject probability.
2.  Then, only if approved, the data cascades to the **Regression model** (Random Forest Regressor). Instead of a yes/no, this outputs a continuous numerical value to dynamically calculate the optimal interest rate based on their exact risk profile.

**Q6: What is SHAP, and why is Explainable AI (XAI) legally and technically important here?**
**Answer:** Advanced algorithms like Random Forests are "Black Boxes"—it's very hard to see their logic. In finance, blocking a loan without an explanation violates transparency regulations (like the GDPR 'Right to Explanation' or the US Equal Credit Opportunity Act). We utilized **SHAP (SHapley Additive exPlanations)**, which leverages cooperative game theory to mathematically calculate the EXACT marginal contribution of every single variable to the final probability, ensuring total transparency.

**Q7: How did you handle Data Preprocessing before feeding it to the algorithm?**
**Answer:** 
1.  **Label Encoding:** ML models only process math. We mapped categorical text (like `RENT` or `HOMEIMPROVEMENT`) into numeric vectors using Scikit-learn's `LabelEncoder`.
2.  **Standard Scaling (Z-Score Normalization):** Variables like "Income" operate in the hundreds of thousands, while "Interest Rate" is a single digit. Without scaling through `StandardScaler`, the ML model would mistakenly think Income is astronomically more important just because the number is bigger.

---

### Category 3: Generative AI & Integration

**Q8: Explain how you integrated Generative AI into a deterministic ML pipeline.**
**Answer:** A deterministic Machine Learning model outputs raw math (e.g., "Probability = 0.82, Debt-to-Income SHAP = +0.15"). We integrated **Google Gemini Flash** purely as a translation and communication layer. We parse the mathematical SHAP output into a structured prompt, and the GenAI acts as an underwriter to dynamically write a plain-English, empathetic summary for the user without interfering with the actual financial mathematics.

**Q9: What happens if the Gemini AI goes down? Does the entire pipeline crash?**
**Answer:** No, the architecture is loosely coupled and highly resilient. If the LLM API is rate-limited or throws a 503 error, our FastAPI backend catches the exception and gracefully generates a fallback predefined statement using the raw probability and predictions. The core classification and regression engines are completely hosted locally and will never go down.
