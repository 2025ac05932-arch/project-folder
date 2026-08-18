import pandas as pd
import streamlit as st
from model.train_models import train_models

st.set_page_config(page_title="Employee Attrition Classification", page_icon="📊", layout="wide")
st.title("📊 Employee Attrition Classification")
st.write("Interactive classification of employee attrition using five machine-learning models.")

@st.cache_resource
def load_models():
    return train_models(save_test_data=False)

try:
    results, trained_models = load_models()
except Exception as e:
    st.error("Application could not train the models.")
    st.exception(e)
    st.stop()

selected_model = st.sidebar.selectbox("Select Classification Model", list(trained_models.keys()))
df = pd.read_csv("Data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
X = df.drop(columns=["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"], errors="ignore")

st.subheader("Enter Employee Information")
input_data = {}
columns = st.columns(3)
for i, feature in enumerate(X.columns):
    with columns[i % 3]:
        if X[feature].dtype == "object":
            options = sorted(X[feature].dropna().unique().tolist())
            input_data[feature] = st.selectbox(feature, options, key=feature)
        else:
            input_data[feature] = st.number_input(feature, min_value=float(X[feature].min()), max_value=float(X[feature].max()), value=float(X[feature].median()), step=1.0 if X[feature].dtype == "int64" else 0.1, key=feature)

if st.button("🔮 Predict Attrition", type="primary"):
    model = trained_models[selected_model]
    input_df = pd.DataFrame([input_data])
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0, 1])
    if prediction == 1:
        st.error(f"### Prediction: Employee Likely to Leave\n\nProbability of Attrition: **{probability:.2%}**")
    else:
        st.success(f"### Prediction: Employee Likely to Stay\n\nProbability of Attrition: **{probability:.2%}**")

st.markdown("---")
st.subheader("Model Performance Comparison")
st.dataframe(results.style.format({c: "{:.4f}" for c in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]}), use_container_width=True)
winner = results.sort_values("F1", ascending=False).iloc[0]
st.info(f"🏆 Overall Winner based on F1 Score: **{winner['ML Model Name']}**")

st.markdown("---")
st.subheader("Dataset Information")
c1, c2, c3 = st.columns(3)
c1.metric("Observations", df.shape[0])
c2.metric("Original Columns", df.shape[1])
c3.metric("Predictive Features", X.shape[1])
st.caption("Dataset: IBM HR Analytics Employee Attrition & Performance – Kaggle")
