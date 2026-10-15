# AI Protein Property Predictor

An AI/ML-based project that predicts whether a human protein is a membrane or non-membrane protein.

## Algorithms
- Logistic Regression
- Random Forest

## Features
- 21 protein features
- Amino acid composition analysis
- Model accuracy, precision, recall and F1-score evaluation

## Technologies
Python, Pandas, NumPy, Scikit-learn and Streamlit.

## Dataset
Uses protein sequence data with membrane/non-membrane labels. The dataset must be added separately.

## Run Locally
```bash
pip install -r requirements.txt
python train_model.py --data data/proteins.csv
streamlit run app.py
```
