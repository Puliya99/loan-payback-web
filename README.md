# Loan Payback Prediction System

A Streamlit web application that predicts the probability of a loan being paid back based on applicant details. The system uses a machine learning model trained on historical loan data.

## 🚀 Overview

This project provides an interactive interface for users to input loan applicant information (income, debt-to-income ratio, credit score, etc.) and receive a risk assessment and probability of payback.

## 🛠 Tech Stack

- **Language:** Python 3.x
- **Web Framework:** [Streamlit]([https://streamlit.io/](https://loan-payback-web-8ghn4xxfgxagbnijwjtmnx.streamlit.app))
- **Data Science & ML:**
  - Pandas
  - NumPy
  - Scikit-learn (Model: Neural Network/Multi-layer Perceptron)
  - Joblib (Model persistence)
- **Package Manager:** `pip`

## 📁 Project Structure

```text
loan-payback-web/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── notebookb9275237dd.ipynb # Model training and exploration notebook
├── input/                  # Dataset files (train.csv, test.csv, etc.)
├── results/                # Trained model artifacts and visualizations
│   ├── loan_payback_nn_model.pkl
│   └── model_columns.pkl
├── loan_payback_nn_model.pkl # Root model artifact (duplicate/backup)
├── model_columns.pkl        # Root columns artifact (duplicate/backup)
└── submission.csv          # Prediction results
```

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## ⚙️ Setup & Run

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/Puliya99/loan-payback-web>
   cd loan-payback-web
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📜 Scripts & Usage

- `streamlit run app.py`: Launches the interactive web dashboard.
- **Model Training:** The model was trained using the provided Jupyter notebook `notebookb9275237dd.ipynb`. To retrain or update the model, open this notebook in a Jupyter environment.

## 🌐 Environment Variables

No specific environment variables are required for the current implementation.

## 🧪 Tests

- [ ] TODO: Implement automated unit and integration tests.

Currently, verification is performed manually by running the Streamlit app and testing various input scenarios.
