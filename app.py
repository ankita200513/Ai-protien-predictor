from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from src.features import extract_features, feature_names

st.set_page_config(page_title="AI Protein Property Predictor", page_icon="🧬")
st.title("AI Protein Property Predictor")
st.write("Classify a protein sequence as membrane or non-membrane using trained models.")
sequence = st.text_area("Enter amino-acid sequence", placeholder="Example: MKLLVIAL...", height=140)
st.caption("Allowed letters: ACDEFGHIKLMNPQRSTVWY")

if st.button("Predict", type="primary"):
    try:
        row = pd.DataFrame([extract_features(sequence)])[feature_names()]
        for filename, title in [
            ("logistic_regression.joblib", "Logistic Regression"),
            ("random_forest.joblib", "Random Forest"),
        ]:
            path = Path("models") / filename
            if not path.exists():
                st.warning(f"{title} model not found. Train the models first using train_model.py.")
                continue
            model = joblib.load(path)
            pred = int(model.predict(row)[0])
            label = "Membrane protein" if pred == 1 else "Non-membrane protein"
            st.subheader(title)
            st.success(f"Prediction: {label}")
            if hasattr(model, "predict_proba"):
                st.write(f"Estimated membrane-class probability: {model.predict_proba(row)[0][1]:.1%}")
    except ValueError as exc:
        st.error(str(exc))

st.info("Educational demo only. Predictions depend on dataset quality and require scientific validation.")
