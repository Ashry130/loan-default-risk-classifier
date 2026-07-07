"""
Load and decode the UCI Statlog (German Credit Data) dataset.
Source: https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
Mirror used: https://raw.githubusercontent.com/jbrownlee/Datasets/master/german.csv
"""
import pandas as pd

COLUMN_NAMES = [
    "checking_account_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_account", "employment_since", "installment_rate_pct",
    "personal_status_sex", "other_debtors_guarantors", "present_residence_since",
    "property", "age_years", "other_installment_plans", "housing",
    "existing_credits_count", "job", "people_liable", "telephone",
    "foreign_worker", "credit_risk"
]

# Category code -> human readable label, per UCI documentation
DECODE_MAPS = {
    "checking_account_status": {
        "A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"
    },
    "credit_history": {
        "A30": "no credits taken", "A31": "all credits paid back duly (this bank)",
        "A32": "existing credits paid back duly", "A33": "delay in past payments",
        "A34": "critical account / other credits existing"
    },
    "purpose": {
        "A40": "new car", "A41": "used car", "A42": "furniture/equipment",
        "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
        "A46": "education", "A47": "vacation", "A48": "retraining",
        "A49": "business", "A410": "other"
    },
    "savings_account": {
        "A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown/no savings account"
    },
    "employment_since": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
        "A74": "4-7 years", "A75": ">= 7 years"
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"
    },
    "other_debtors_guarantors": {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor"
    },
    "property": {
        "A121": "real estate", "A122": "building society savings/life insurance",
        "A123": "car or other", "A124": "unknown/no property"
    },
    "other_installment_plans": {
        "A141": "bank", "A142": "stores", "A143": "none"
    },
    "housing": {
        "A151": "rent", "A152": "own", "A153": "for free"
    },
    "job": {
        "A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified"
    },
    "telephone": {
        "A191": "none", "A192": "yes, registered"
    },
    "foreign_worker": {
        "A201": "yes", "A202": "no"
    },
}


def load_raw(path="data/german_credit.csv") -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df


def load_decoded(path="data/german_credit.csv") -> pd.DataFrame:
    df = load_raw(path)
    for col, mapping in DECODE_MAPS.items():
        df[col] = df[col].map(mapping)
    # Original encoding: 1 = good credit risk, 2 = bad credit risk.
    # Recode to standard convention: 1 = bad/default risk (positive class), 0 = good.
    df["default_risk"] = df["credit_risk"].map({1: 0, 2: 1})
    df = df.drop(columns=["credit_risk"])
    return df


if __name__ == "__main__":
    df = load_decoded()
    print(df.shape)
    print(df.head())
    print(df["default_risk"].value_counts(normalize=True))
