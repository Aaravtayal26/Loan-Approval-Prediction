# Loan Approval Prediction Interactive App

A Streamlit app that trains and compares Random Forest, Logistic Regression, and SVC using the supplied loan dataset.

## Run in VS Code
1. Extract this folder and open it in VS Code.
2. Make sure Python is installed.
3. Open VS Code Terminal (`Terminal > New Terminal`).
4. Create/activate a virtual environment (optional but recommended):
   - Windows: `py -m venv .venv`
   - Windows PowerShell: `.venv\\Scripts\\Activate.ps1`
5. Install dependencies: `pip install -r requirements.txt`
6. Start the app: `streamlit run app.py`
7. Streamlit opens the app in your browser (usually at `http://localhost:8501`).

The included CSV is loaded automatically. You may also upload another CSV from the sidebar, provided it has the same feature and target columns.


## Pages
- Dashboard: dataset overview and performance snapshot
- Loan Prediction: single-model prediction form
- Compare Models: predictions from all three models on the same applicant
- Model Analytics: evaluation metrics, confusion matrix, and ROC curves
- About: project description


## Important
This is an academic demonstration only, not a real lending decision tool. It retrains models when the dataset is first loaded in a session. It does not store entered applicant details.
