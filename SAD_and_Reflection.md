# SAD Mini-Analysis, Evaluation Results and Reflection

## 1. SAD mini-analysis
| Element | Response |
|---|---|
| Problem statement | Academic risk is often noticed only after poor results. The prototype explores early identification of students at risk of failing Mathematics (G3 < 10) using indicators that already exist before the final grade. |
| User | Faculty member or academic analyst who wants to flag learners who may need extra support. |
| Inputs | A CSV of student records: 30 demographic, social and school attributes (e.g. studytime, absences, failures, Medu/Fedu, goout) plus the period grades G1 and G2. G3 is never an input. |
| Process | Upload CSV → check required columns → impute, scale and encode → Random Forest predicts risk → display and export results. |
| Outputs | AT RISK / NOT AT RISK label per student, count of predicted at-risk records, downloadable results CSV, and a model evaluation summary (accuracy, precision, recall, F1). |
| Key limitation | Trained on one Portuguese secondary-school dataset (395 records); it may not generalise to another institution or population, and it is not a basis for real decisions on its own. |

## 2. Model evaluation results
Fill in after running `python train_model.py` (20% stratified test set, random_state=42).

| Metric | Value |
|---|---|
| Accuracy | ____ |
| Precision (At risk) | ____ |
| Recall (At risk) | ____ |
| F1-score (At risk) | ____ |

Confusion matrix: ____

## 3. Testing evidence
| Test | Expected result | Pass? |
|---|---|---|
| Run train_model.py | CSV and model files created without errors | ☐ |
| Run Streamlit app | Dashboard loads in browser | ☐ |
| Upload sample CSV | Predictions are displayed | ☐ |
| Upload CSV missing a required column | Helpful error message | ☐ |
| Download results | CSV containing predictions downloads | ☐ |
| Open GitHub repository | Source files and README visible | ☐ |

## 4. Reflection
1. **Prediction target and why G3 is excluded.** The target is a binary label: AT RISK if the final Mathematics grade G3 < 10, otherwise NOT AT RISK. G3 is excluded from the features because the label is computed directly from it. Including it would be target leakage: the model would simply learn the cutoff, score near-perfectly, and be useless for predicting a student's risk before the final grade exists.
2. **What recall for the "At risk" class tells you.** Recall is the share of students who actually failed that the model caught. A recall of, say, 0.80 means about 1 in 5 truly at-risk students is missed. For a support-flagging tool, missed students are the costly error, so recall for At risk matters more than overall accuracy, which can look high just because most students pass.
3. **Why it may perform differently at my institution.** The data comes from two Portuguese secondary schools in one period. Grading scales, curriculum, student demographics, family and school context, and the meaning of attributes like absences or failures differ elsewhere, so the learned patterns (data and concept shift) may not hold. With only 395 records, the metrics are also a noisy estimate. The model should be validated on local data before use.
4. **What a faculty member should do after an AT RISK prediction.** Treat it as a prompt, not a verdict: review the student's actual record, talk with the student, and look for causes (attendance, workload, personal circumstances). Offer support such as tutoring or consultation, monitor progress, and never use the flag alone for grading, penalties or placement decisions.
