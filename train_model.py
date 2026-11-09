# train_model.py
"""Download the UCI Student Performance (Mathematics) data, train a Random Forest
to flag students at risk of failing (G3 < 10), and save the model.

Usage:
    python train_model.py                 # download from UCI, then train
    python train_model.py --csv my.csv    # skip download, use an existing CSV
"""
import argparse
import io
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MATH_ROWS = 395  # the Mathematics file has 395 records (Portuguese has 649)
ZIP_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


def download_data() -> pd.DataFrame:
    """Section 1: retrieve the dataset (ucimlrepo, with a direct-zip fallback)."""
    from ucimlrepo import fetch_ucirepo

    student = fetch_ucirepo(id=320)
    df = pd.concat([student.data.features, student.data.targets], axis=1)
    if len(df) != MATH_ROWS:
        # ucimlrepo may return the Portuguese-course records; the lab needs Maths.
        print(f"ucimlrepo returned {len(df)} rows, expected {MATH_ROWS} (Maths). "
              "Falling back to student-mat.csv from the UCI zip file.")
        raw = urllib.request.urlopen(ZIP_URL, timeout=60).read()
        outer = zipfile.ZipFile(io.BytesIO(raw))
        inner = zipfile.ZipFile(io.BytesIO(outer.read("student.zip")))
        df = pd.read_csv(inner.open("student-mat.csv"), sep=";")
    return df


def main(csv_path: str | None) -> None:
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = download_data()
        df.to_csv("student_data.csv", index=False)

    print("Rows and columns:", df.shape)
    print(df.head())
    print(df.dtypes)
    print("Missing values per column:")
    print(df.isnull().sum())

    # ---- Section 2: preprocess, train, evaluate ----
    # Target: 1 = at risk (G3 < 10); 0 = not at risk
    df = df.drop_duplicates()
    y = (df["G3"] < 10).astype(int)

    # Never use G3 as a predictor (target leakage)
    X = df.drop(columns=["G3"])
    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nTest set size:", len(y_test), "| at-risk in test set:", int(y_test.sum()))
    print("Accuracy :", round(accuracy_score(y_test, predictions), 3))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 3), "(At risk)")
    print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 3), "(At risk)")
    print("F1-score :", round(f1_score(y_test, predictions, zero_division=0), 3), "(At risk)")
    print("\nConfusion matrix [rows=actual, cols=predicted] (Not at risk, At risk):")
    print(confusion_matrix(y_test, predictions))
    print()
    print(classification_report(
        y_test, predictions,
        target_names=["Not at risk", "At risk"], zero_division=0))

    joblib.dump(model, "student_model.pkl")
    print("Saved student_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="use this CSV instead of downloading from UCI")
    main(parser.parse_args().csv)
