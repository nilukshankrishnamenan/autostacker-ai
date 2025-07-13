import streamlit as st
import pandas as pd
from autostacker.runner import train_and_evaluate_models
from autostacker.gemini_helper import get_gemini_explanation

st.set_page_config(page_title="AutoStacker AI", layout="wide")

st.title("🤖 AutoStacker AI")
st.markdown("Upload a CSV and let Gemini + AutoML evaluate the best model for your data.")

uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("🎯 Select your target column", df.columns)
    lang = st.selectbox("🌐 Choose explanation language", ["English", "Sinhala", "Tamil"])

    if st.button("🚀 Train and Evaluate"):
        with st.spinner("Training models..."):
            X = df.drop(columns=[target_column])
            y = df[target_column]

            results = train_and_evaluate_models(X, y)
            st.success("✅ Training complete!")

            st.subheader("📈 Model Performance")
            st.json(results)

            st.subheader("💡 Gemini Explanation")
            explanation = get_gemini_explanation(results, language=lang)
            st.markdown(explanation)
else:
    st.info("Please upload a CSV file to begin.")
