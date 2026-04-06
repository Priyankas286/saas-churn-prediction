"""
src/features/build_features.py
-------------------------------
Takes raw CSV → clean, encoded, feature-engineered DataFrame
ready for model training.

WHAT WE DO HERE:
1. Drop columns the model doesn't need (customer_id)
2. Create new smart features from existing ones
3. Encode categorical columns (plan_type, payment_method)
4. Scale numerical columns so the model treats them fairly
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle, os

def load_raw(path: str) -> pd.DataFrame:
    """Load raw CSV and do basic sanity checks."""
    df = pd.read_csv(path)
    print(f"[load]  Shape: {df.shape}  |  Churn rate: {df['churned'].mean():.1%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that help the model understand customer health.
    These are inspired by real SaaS Product Analytics thinking.
    """
    df = df.copy()

    # 1. Revenue-at-risk per user seat
    #    A high-paying account with many users matters more to the business
    df["revenue_per_user"] = df["monthly_charges"] / (df["num_users"] + 1)

    # 2. Support burden score
    #    Customers who raise many tickets relative to tenure are frustrated
    df["support_intensity"] = df["num_support_tickets"] / (df["tenure_months"] + 1)

    # 3. Engagement score (0–100, higher = more engaged)
    #    Combines feature adoption and recency of login
    login_score    = 1 - (df["last_login_days_ago"] / 90)          # 1 = logged in today
    login_score    = login_score.clip(0, 1)
    adoption_score = df["feature_adoption_pct"] / 100
    df["engagement_score"] = ((login_score * 0.5) + (adoption_score * 0.5)) * 100

    # 4. Customer lifetime value proxy
    df["ltv_proxy"] = df["total_charges"] / (df["tenure_months"] + 1)

    # 5. Is the customer "at-risk" by simple rule? (useful signal)
    df["at_risk_flag"] = (
        (df["last_login_days_ago"] > 30) &
        (df["feature_adoption_pct"] < 30)
    ).astype(int)

    print(f"[features]  Added 5 engineered features. New shape: {df.shape}")
    return df


def encode_and_scale(df: pd.DataFrame, fit: bool = True,
                     encoders_path: str = "src/models/encoders.pkl") -> pd.DataFrame:
    """
    Encode categoricals + scale numerics.

    fit=True  → learn the encoding (training time)
    fit=False → apply saved encoding (prediction time)
    """
    df = df.copy()
    df.drop(columns=["customer_id"], errors="ignore", inplace=True)

    CAT_COLS = ["plan_type", "payment_method"]
    NUM_COLS = [c for c in df.columns if c not in CAT_COLS + ["churned"]]

    if fit:
        encoders = {}
        for col in CAT_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        scaler = StandardScaler()
        df[NUM_COLS] = scaler.fit_transform(df[NUM_COLS])
        encoders["scaler"] = scaler
        encoders["num_cols"] = NUM_COLS

        os.makedirs(os.path.dirname(encoders_path), exist_ok=True)
        with open(encoders_path, "wb") as f:
            pickle.dump(encoders, f)
        print(f"[encode]  Encoders saved → {encoders_path}")

    else:
        with open(encoders_path, "rb") as f:
            encoders = pickle.load(f)
        for col in CAT_COLS:
            df[col] = encoders[col].transform(df[col])
        df[encoders["num_cols"]] = encoders["scaler"].transform(df[encoders["num_cols"]])
        print(f"[encode]  Encoders loaded from {encoders_path}")

    return df


def build_features(raw_path: str, out_path: str = "data/features.csv") -> pd.DataFrame:
    """Full pipeline: raw CSV → feature CSV."""
    df = load_raw(raw_path)
    df = engineer_features(df)
    df = encode_and_scale(df, fit=True)
    df.to_csv(out_path, index=False)
    print(f"[done]  Features saved → {out_path}")
    return df


if __name__ == "__main__":
    build_features("data/saas_churn.csv")
