# Student Performance Prediction System

A classroom prototype (Data Mining laboratory) that estimates whether a secondary-school
student is **at risk of failing the final Mathematics grade**, with a Streamlit interface
for uploading student records and downloading predictions.

## Problem
Academic risk is often noticed only after poor results. This prototype explores early
identification of students who may need extra academic support, using historical records.

## Dataset
[UCI Machine Learning Repository – Student Performance (ID 320)](https://archive.ics.uci.edu/dataset/320/student+performance),
Mathematics records (395 students, 30 predictor attributes plus G1, G2, G3).

## Prediction target
`G3 < 10` → **AT RISK** (1); `G3 ≥ 10` → **NOT AT RISK** (0).
`G3` is the target only and is never used as an input feature. `G1` and `G2`
(earlier period grades) are used as inputs.

## Technologies
Python 3.10+, pandas, scikit-learn (Random Forest pipeline), joblib, Streamlit, ucimlrepo.

## Installation and run
```bash
git clone https://github.com/YOUR-USERNAME/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
python train_model.py        # downloads data, trains, prints metrics, saves student_model.pkl
streamlit run app.py         # opens http://localhost:8501
```
In the app, download the sample input CSV, upload it, and review/download the predictions.

## Model evaluation
Run `python train_model.py` to print accuracy, precision, recall and F1-score for the
held-out 20% test set (stratified, `random_state=42`). My results are recorded in
[SAD_and_Reflection.md](SAD_and_Reflection.md).

## Limitations
- Trained on one Portuguese secondary-school dataset; it may not generalise to other
  institutions, curricula or populations.
- Small dataset (395 records), so metrics vary with the train/test split.
- G1 and G2 are strong predictors, so predictions are only available once those grades exist.
- Predictions are not diagnoses and must not be the sole basis for real student decisions.
  An "AT RISK" flag should prompt a supportive conversation, not a penalty.
