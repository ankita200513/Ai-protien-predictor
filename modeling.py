import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features import extract_features, feature_names

def prepare_dataset(df):
    missing = {"sequence", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    X = pd.DataFrame([extract_features(s) for s in df["sequence"]])[feature_names()]
    y = pd.to_numeric(df["label"], errors="raise").astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("label must contain both classes: 0 and 1.")
    return X, y

def train_evaluate(df, test_size=0.2, random_state=42):
    X, y = prepare_dataset(df)
    if "pfam_family" in df.columns and df["pfam_family"].notna().all():
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=df["pfam_family"].astype(str)))
        split_name = "GroupShuffleSplit by pfam_family"
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            raise ValueError("Group split produced a one-class train/test set; check family distribution.")
    else:
        train_idx, test_idx = train_test_split(range(len(df)), test_size=test_size,
                                               random_state=random_state, stratify=y)
        split_name = "Stratified random split (pfam_family unavailable)"
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state))
        ]),
        "random_forest": RandomForestClassifier(n_estimators=300,
            class_weight="balanced_subsample", random_state=random_state, n_jobs=-1)
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_membrane": float(precision_score(y_test, pred, zero_division=0)),
            "recall_membrane": float(recall_score(y_test, pred, zero_division=0)),
            "f1_membrane": float(f1_score(y_test, pred, zero_division=0)),
            "test_records": int(len(test_idx))
        }
    return models, results, split_name

def save_artifacts(models, results, split_name, output_dir="models"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, out / f"{name}.joblib")
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"split": split_name, "results": results}, f, indent=2)
