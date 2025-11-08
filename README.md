💳 Credit Card Fraud Detection Dashboard

A Streamlit-based web dashboard for detecting potential fraudulent credit card transactions using a pre-trained machine learning model. 
The app provides real-time fraud predictions, visual metrics, downloadable reports, and alerts for high-risk transactions.

Project Overview

This project allows users (banks, analysts, or individuals) to upload transaction data in CSV format and instantly view fraud detection results.

It integrates a trained ML model (fraud_model.pkl) and provides:
Predictions for each transaction (fraudulent or legitimate)
Fraud probability scores
Confusion matrix & classification metrics
PDF and CSV report downloads
Fraud notification simulation

Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
2. Install Dependencies
pip install -r requirements.txt
3. Place Model File
Make sure the trained model file fraud_model.pkl is in the project root directory.
If you don’t have one, train it using your ML notebook or script.
4. Run the App
streamlit run app.py

📂 File Structure
.
├── app.py                     # Main Streamlit app
├── fraud_model.pkl            # Trained model
├── requirements.txt           # Dependencies
├── data/                      # (Optional) Sample CSV files
├── reports/                   # Auto-generated PDF reports
└── README.md

📤 Usage Instructions
🧾 1. Upload Transactions

Click “Upload CSV File”

The app displays the first 10 rows for review

🔍 2. View Predictions

Each row shows:

Prediction: 1 = Fraud, 0 = Legitimate

Fraud Probability: Likelihood score

Fraudulent transactions are highlighted in red

📊 3. Check Metrics

If the uploaded CSV contains a Class column (actual labels):

The app displays:

F1 Score

AUC-ROC Score

Confusion Matrix Heatmap

Detailed Classification Report

💾 4. Download Reports

Download Results (CSV): fraud_predictions.csv

Generate PDF Summary: Includes fraud counts and top 10 suspicious transactions

🧩 Features

✅ Real-time prediction using a pre-trained model
✅ Highlights suspicious transactions in red
✅ Shows key performance metrics (F1, AUC, confusion matrix)
✅ Downloadable CSV and PDF summary reports
✅ Simulated fraud alert system

# PDF Report Example:

The generated PDF includes:
Summary counts
Fraud alert notice
Top 10 flagged transactions
Fraud reporting link

🔒 Security Notes

Model runs locally (no data is sent to a server)
Uploaded data is processed in memory and not stored
Safe for testing on sensitive transaction data
