# Financial Risk Analytics & ML Pipeline

This project is an end-to-end Machine Learning pipeline for predicting loan default risk. It features a modern, interactive Streamlit frontend and a robust FastAPI backend. It also includes an AI Agent (powered by Google Gemini) to provide personalized financial advice based on the model's risk assessment.

## Project Structure

- `src/api.py`: FastAPI backend application.
- `app.py`: Streamlit frontend dashboard.
- `models/`: Trained machine learning models and encoders.
- `notebooks/`: Jupyter notebooks for data exploration and model training.
- `docker-compose.yml`: Configuration for running the project with Docker.
- `Dockerfile.api` & `Dockerfile.frontend`: Docker images setup.

---

## 🚀 How to Run the Project (Step-by-Step)

You can run this project locally using Python or via Docker. 

### Method 1: Running Locally (Python)

**Step 1: Create a Virtual Environment (Optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

**Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3: Set up Environment Variables**
Create a `.env` file in the root directory and add your Google Gemini API key:
```env
GEMINI_API_KEY=your_genai_api_key_here
```

**Step 4: Start the FastAPI Backend**
Open a terminal and run the backend server:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```
*The API will be available at `http://localhost:8000` (Swagger UI at `/docs`).*

**Step 5: Start the Streamlit Frontend**
Open a **new** terminal window/tab, navigate to the project directory, and run:
```bash
streamlit run app.py
```
*The Streamlit App will automatically open in your browser at `http://localhost:8501`.*

---

### Method 2: Running with Docker

**Step 1: Ensure Docker is Running**
Make sure the Docker daemon/Docker Desktop is installed and running on your system.

**Step 2: Build and Start Containers**
Run the following command in the root of the project:
```bash
docker-compose up --build -d
```
*Wait a few moments for the images to build and containers to start.*

**Step 3: Access the Application**
- **Frontend Dashboard:** `http://localhost:8501`
- **Backend API Docs:** `http://localhost:8000/docs`

**Step 4: Stop the Application**
When you're done, you can stop the containers with:
```bash
docker-compose down
```

---

## 🌟 Advanced Enterprise Features
- **FastAPI Native Backend:** High-performance REST API.
- **Cascading ML Architecture:** Random Forest predicts approval probability, and a separate XGBoost/Random Forest Regression model dynamically calculates the Optimal Interest Rate.
- **Explainable AI (SHAP):** Game-theoretic Mathematical Explainability predicting EXACTLY why a loan was approved or rejected (feature by feature).
- **Interactive "What-If" Simulation:** Real-time XAI Sandbox dynamically recalculating mathematical SHAP risk based on simulated financial slider inputs.
- **Generative XAI Agent:** Automated NLP Underwriter Notes using Google Gemini Flash contextually woven with SHAP metrics.
- **Automated Official PDF Generation:** Dynamic PDF Generator exporting XAI decision logic and Applicant Profiles instantly via Streamlit.
- **Modern UI:** Premium, responsive dashboard built with advanced XAI Plotly charts.
