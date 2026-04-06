"""
generate_data.py
----------------
Generates a realistic synthetic SaaS customer churn dataset.
Run this once to create: data/saas_churn.csv

What each column means:
- customer_id         : Unique ID for each customer
- tenure_months       : How long they have been subscribed
- monthly_charges     : What they pay per month (INR/USD)
- total_charges       : Total amount paid since joining
- num_users           : Number of team seats on the account
- num_support_tickets : Support tickets raised in last 90 days
- last_login_days_ago : Days since they last logged into the app
- feature_adoption    : % of product features they actively use (0–100)
- plan_type           : Free, Basic, Pro, or Enterprise
- payment_method      : Card, UPI, Net Banking, Invoice
- churned             : 1 = customer left, 0 = still active  ← TARGET
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000  # number of customers

# ── Base features ──────────────────────────────────────────
tenure          = np.random.randint(1, 60, N)
monthly_charges = np.round(np.random.uniform(500, 5000, N), 2)
num_users       = np.random.randint(1, 50, N)
support_tickets = np.random.poisson(lam=2, size=N)
last_login      = np.random.randint(0, 90, N)
feature_adopt   = np.round(np.random.uniform(5, 100, N), 1)

plan_type       = np.random.choice(["Free", "Basic", "Pro", "Enterprise"],
                                    N, p=[0.3, 0.35, 0.25, 0.10])
payment_method  = np.random.choice(["Card", "UPI", "Net Banking", "Invoice"],
                                    N, p=[0.40, 0.30, 0.20, 0.10])

# ── Churn logic (realistic business rules) ─────────────────
# High churn risk factors:
#   - Short tenure
#   - Low feature adoption
#   - Many support tickets
#   - Long time since last login
#   - Free plan

churn_score = (
    (tenure < 6).astype(int) * 0.25 +
    (feature_adopt < 30).astype(int) * 0.30 +
    (support_tickets > 4).astype(int) * 0.20 +
    (last_login > 30).astype(int) * 0.15 +
    (plan_type == "Free").astype(int) * 0.10 +
    np.random.uniform(0, 0.15, N)           # noise
)

churned = (churn_score > 0.45).astype(int)

# ── Assemble DataFrame ──────────────────────────────────────
df = pd.DataFrame({
    "customer_id"          : [f"CUST_{i:05d}" for i in range(N)],
    "tenure_months"        : tenure,
    "monthly_charges"      : monthly_charges,
    "total_charges"        : np.round(tenure * monthly_charges * np.random.uniform(0.8, 1.0, N), 2),
    "num_users"            : num_users,
    "num_support_tickets"  : support_tickets,
    "last_login_days_ago"  : last_login,
    "feature_adoption_pct" : feature_adopt,
    "plan_type"            : plan_type,
    "payment_method"       : payment_method,
    "churned"              : churned
})

df.to_csv("data/saas_churn.csv", index=False)
print(f"✅ Dataset saved → data/saas_churn.csv")
print(f"   Rows : {len(df)}")
print(f"   Churn rate : {df['churned'].mean():.1%}")
print(df.head(3).to_string())
