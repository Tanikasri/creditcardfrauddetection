import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("This app uses a trained Machine Learning model to detect potential fraudulent transactions in real-time.")

# ------------------------------------------------------------
# Load the Model
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

model = load_model()

# ------------------------------------------------------------
# Rename Columns (Human-friendly)
# ------------------------------------------------------------
def rename_columns_readable(df):
    rename_dict = {
        "Time": "Transaction_Time",
        "Amount": "Transaction_Amount",
        "Class": "Is_Fraud",
        "V1": "Transaction Amount Deviation",
        "V2": "Merchant Risk Index",
        "V3": "Card Usage Frequency",
        "V4": "Account Age Score",
        "V5": "Device Trust Level",
        "V6": "Geolocation Risk Factor",
        "V7": "Past Fraud Likelihood",
        "V8": "Transaction Velocity",
        "V9": "Customer Spending Pattern",
        "V10": "Authentication Confidence",
        "V11": "Credit Utilization Rate",
        "V12": "Merchant Type Influence",
        "V13": "Purchase Category Risk",
        "V14": "Customer Income Sensitivity",
        "V15": "Transaction Time Variance",
        "V16": "Account Behaviour Drift",
        "V17": "Cardholder Verification Score",
        "V18": "Transaction Channel Risk",
        "V19": "Recurring Payment Indicator",
        "V20": "Unusual Spending Spike",
        "V21": "Customer Loyalty Factor",
        "V22": "Merchant Cluster ID",
        "V23": "Cross-Border Indicator",
        "V24": "Transaction Amount Ratio",
        "V25": "Card Reissue Frequency",
        "V26": "Device Fingerprint Variation",
        "V27": "Account Login Deviation",
        "V28": "Transaction Location Mismatch"
    }
    return df.rename(columns=rename_dict)

# ------------------------------------------------------------
# Prediction Function
# ------------------------------------------------------------
def get_prediction_scores(model, X):
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = preds
    return preds, scores

# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file of transactions", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    df_original = df.copy()
    df_display = rename_columns_readable(df.copy())

    st.markdown("### 📊 Sample of Uploaded Data")
    st.dataframe(df_display.head(10))

    # -----------------------------------------------
    # Predictions
    # -----------------------------------------------
    features = df_original.drop("Class", axis=1, errors="ignore")
    preds, scores = get_prediction_scores(model, features)
    df_display["Prediction"] = preds
    df_display["Fraud Probability"] = scores

    # -----------------------------------------------
    # Fraud Summary
    # -----------------------------------------------
    total = len(df_display)
    frauds = (preds == 1).sum()
    legits = total - frauds

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Legitimate Transactions", f"{legits:,}")
    col3.metric("Fraudulent Transactions", f"{frauds:,}")

    # -----------------------------------------------
    # Highlight Fraudulent Transactions
    # -----------------------------------------------
    def highlight_fraud(row):
        return ["background-color: red; color: white" if row["Prediction"] == 1 else "" for _ in row]

    st.markdown("### 🔍 Detected Transactions")
    if df_display.size > 262144:
        st.warning("⚠️ Dataset too large to highlight — showing first 1000 rows.")
        st.dataframe(df_display.head(1000))
    else:
        st.dataframe(df_display.style.apply(highlight_fraud, axis=1))

    # -----------------------------------------------
    # Model Metrics
    # -----------------------------------------------
    if "Class" in df_original.columns:
        y_true = df_original["Class"]
        f1 = f1_score(y_true, preds)
        auc = roc_auc_score(y_true, scores)

        st.markdown("### 📈 Model Performance Metrics")
        st.write(f"**F1 Score:** {f1:.3f}")
        st.write(f"**AUC-ROC Score:** {auc:.3f}")

        # Confusion Matrix
        cm = confusion_matrix(y_true, preds)
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        report = classification_report(y_true, preds, output_dict=False)
        st.markdown("#### Detailed Classification Report")
        st.text(report)

    # -----------------------------------------------
    # Downloadable CSV
    # -----------------------------------------------
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", csv, "fraud_predictions.csv", "text/csv")

    # -----------------------------------------------
    # Notification Simulation
    # -----------------------------------------------
    st.markdown("### 🔔 Notification Simulation")
    if frauds > 0:
        st.error(f"⚠️ {frauds} suspicious transactions detected! Notification sent to Bank & Customer Service.")
    else:
        st.success("✅ No fraudulent transactions detected. All clear.")

    # -----------------------------------------------
    # Generate and Download PDF Report
    # -----------------------------------------------
    st.markdown("### 🧾 Generate Fraud Summary Report")

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    text_object = c.beginText(50, 800)
    text_object.setFont("Helvetica", 11)

    text_lines = [
        "💳 Credit Card Fraud Detection Report",
        "",
        f"Total Transactions: {total}",
        f"Legitimate Transactions: {legits}",
        f"Fraudulent Transactions: {frauds}",
        "",
    ]

    if frauds > 0:
        text_lines.append("⚠️ Fraudulent Activity Detected.")
        text_lines.append("Please review the flagged transactions immediately.")
        text_lines.append("🔗 Report Fraudulent Activity: https://securebankalerts.com/report-fraud")
        text_lines.append("")

        # Show a few sample fraudulent transactions
        sample_frauds = df_display[df_display["Prediction"] == 1].head(10)
        text_lines.append("■■ Flagged Fraud Transactions (Sample)")
        for _, row in sample_frauds.iterrows():
            text_lines.append(
                f"- Time: {row.get('Transaction_Time', 'N/A')} | "
                f"Amount: {row.get('Transaction_Amount', 'N/A')} | "
                f"Prob: {row.get('Fraud Probability', 0):.2f}"
            )
    else:
        text_lines.append("✅ No fraudulent transactions detected.")

    for line in text_lines:
        text_object.textLine(line)

    c.drawText(text_object)
    c.showPage()
    c.save()

    pdf_bytes = buffer.getvalue()
    buffer.close()

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="fraud_detection_report.pdf",
        mime="application/pdf"
    )
