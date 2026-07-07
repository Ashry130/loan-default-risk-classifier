# Loan Default Risk Classifier

A cost-sensitive machine learning model for consumer loan approval decisions, built as a
portfolio case study in fintech/banking credit risk analytics.

## The problem

When a lender evaluates a loan application, two kinds of mistakes are possible — and they are
**not equally costly**:

- Rejecting a good applicant → lost interest income
- Approving a bad applicant → potential loss of principal (typically far more expensive)

Most tutorial-level classifiers optimize for accuracy, which silently assumes both mistakes cost
the same. This project builds a model that is explicitly evaluated against a **business cost
metric**, not just accuracy — the same framing used in real credit risk scoring work.

This approach mirrors work I did during a Data Analytics internship at Banque Misr's Digital,
AI & Data Governance department, where I built a loan approval classifier and engineered
transaction-based risk features. That work used proprietary bank data and can't be shared here —
this repo rebuilds the same methodology on a public dataset so it can be shown openly.

## Dataset

[UCI Statlog (German Credit Data)](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29) —
1,000 loan applicants, 20 attributes (checking account status, credit history, loan purpose,
amount, duration, employment, demographics, etc.), labeled good/bad credit risk. Class balance is
70% good / 30% bad, which is handled explicitly in both training and evaluation.

## Approach

1. **EDA** — class balance, numeric feature distributions by risk class, categorical feature
   default rates
2. **Feature engineering** — one-hot encoding for categoricals, standard scaling for numerics,
   wrapped in a single scikit-learn `Pipeline` to prevent train/test leakage
3. **Models compared** — Logistic Regression (interpretable baseline, common in regulated lending
   contexts), Random Forest, XGBoost — all trained with class-imbalance handling
4. **Evaluation** — ROC-AUC, precision/recall on the minority (bad-risk) class, and a **custom
   business cost metric** (5x cost for approving a bad-risk applicant vs. rejecting a good one,
   per the dataset's documented cost matrix)
5. **Feature importance** — which factors the winning model actually relies on, checked against
   domain intuition

## Key result

The model with the best ROC-AUC (XGBoost, 0.809) was **not** the best business choice.
Logistic Regression, despite a marginally lower AUC (0.808), had substantially higher recall on
risky applicants (78.7% vs 61.3%) and came out **~30% cheaper** under the business cost metric.

| Model | ROC-AUC | Recall (bad risk) | Precision (bad risk) | Business cost |
|---|---|---|---|---|
| **Logistic Regression** | 0.808 | 0.787 | 0.557 | **127** |
| XGBoost | 0.809 | 0.613 | 0.575 | 179 |
| Random Forest | 0.800 | 0.427 | 0.696 | 229 |

This is the central takeaway I'd bring to a lending stakeholder: **the metric you optimize for
changes which model you should ship.** A data science team chasing AUC would have picked the
wrong model for the business.

## Repo structure

```
loan-risk-classifier/
├── data/
│   └── german_credit.csv          # raw UCI dataset
├── src/
│   └── load_data.py               # loading + category decoding
├── loan_default_risk_classifier.ipynb   # full analysis, executed with outputs
├── requirements.txt
└── README.md
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook loan_default_risk_classifier.ipynb
```

## About this project

Built by Mikey, a Computer Science / Data Analytics student, as part of a portfolio focused on
fintech and banking data analytics — risk scoring, fraud/anomaly detection, and customer
segmentation. Open to freelance work in this space.
