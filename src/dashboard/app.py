import streamlit as st
from explainai.explain import compute_shap_values
from explainai.models.example_model import train_example_model

import pandas as pd


st.title("ExplainAI Dashboard")

st.write("Train example model…")
model, X_test, feature_names = train_example_model()

df = pd.DataFrame(X_test, columns=feature_names)

if st.button("Compute SHAP"):
    shap_vals = compute_shap_values(model, df)
    st.write("SHAP Values:")
    st.dataframe(shap_vals)
