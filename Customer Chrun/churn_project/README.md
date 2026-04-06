# 🔮 SaaS Customer Churn Prediction — End-to-End ML Pipeline

> **Predict which customers will cancel before they do** — and give the business time to act.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-API-green?logo=flask)](https://flask.palletsprojects.com)
[![Airflow](https://img.shields.io/badge/Airflow-DAG-red?logo=apacheairflow)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Business Problem

For any SaaS company, **losing a customer (churn) is 5–7× more expensive than retaining one.**

This project answers: *"Which of our customers is most likely to cancel in the next 30 days — and what should we do about it?"*

The output is a live REST API that Customer Success teams can query to get:
- Churn probability score (0–100%)
- Risk level: LOW / MEDIUM / HIGH
- Business recommendation (e.g. "Assign CSM + offer 20% discount")

---

## 🏗️ Architecture

```
Raw Data (CSV / S3)
        │
        ▼
┌───────────────────┐
│  Airflow DAG      │  ← Orchestrates the full pipeline on a schedule
│  (churn_pipeline) │
└───────┬───────────┘
        │
   ┌────▼────┐    ┌──────────────────┐    ┌──────────────────┐
   │ Ingest  │───▶│ Feature Engineer │───▶│  Train Models    │
   │  Data   │    │ (5 new features) │    │  LR / RF / GB    │
   └─────────┘    └──────────────────┘    └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  MLflow Tracking  │
                                          │  (params+metrics) │
                                          └────────┬─────────┘
                                                   │
                                          ┌────────▼─────────┐
                                          │  Flask REST API   │
                                          │  /predict         │
                                          └──────────────────┘
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.81 | ~0.72 | ~0.68 | ~0.70 | ~0.87 |
| **Random Forest** ✅ | **~0.88** | **~0.82** | **~0.76** | **~0.79** | **~0.93** |
| Gradient Boosting | ~0.87 | ~0.80 | ~0.74 | ~0.77 | ~0.92 |

> **Why Recall matters here:** A false negative (predicting a customer stays but they leave) costs real revenue. We prioritise Recall — catch as many churners as possible.

---

## 🔬 Features Used

### Raw features
| Feature | Description |
|---|---|
| `tenure_months` | How long the customer has been subscribed |
| `monthly_charges` | Monthly subscription amount |
| `num_support_tickets` | Support tickets in last 90 days |
| `last_login_days_ago` | Days since last app login |
| `feature_adoption_pct` | % of product features actively used |
| `plan_type` | Free / Basic / Pro / Enterprise |

### Engineered features (created in `build_features.py`)
| Feature | Business Logic |
|---|---|
| `engagement_score` | Weighted score of login recency + feature adoption |
| `support_intensity` | Support tickets per month of tenure (frustration signal) |
| `revenue_per_user` | Monthly charges ÷ seat count (account value signal) |
| `ltv_proxy` | Total revenue ÷ tenure (lifetime value estimate) |
| `at_risk_flag` | Binary: low adoption AND inactive > 30 days |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/priyankas286/saas-churn-prediction.git
cd saas-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (one command)
```bash
python dags/churn_pipeline_dag.py
```
This runs all 4 stages:
- ✅ Data ingestion
- ✅ Feature engineering
- ✅ Model training + MLflow logging
- ✅ Model saved, ready to serve

### 4. Launch the prediction API
```bash
python app/app.py
```
Open `http://localhost:5000` in your browser.

### 5. Make a prediction
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure_months": 3,
    "monthly_charges": 1200,
    "total_charges": 3600,
    "num_users": 2,
    "num_support_tickets": 6,
    "last_login_days_ago": 45,
    "feature_adoption_pct": 15,
    "plan_type": "Free",
    "payment_method": "UPI"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.8342,
  "churn_prediction": 1,
  "risk_level": "HIGH",
  "recommendation": "🚨 HIGH RISK: Assign a Customer Success Manager immediately. Offer a 25% discount on renewal. Schedule a product review call within 48 hours.",
  "model_used": "Random Forest",
  "model_roc_auc": 0.9287
}
```

---

## 📁 Project Structure

```
saas-churn-prediction/
│
├── data/
│   ├── generate_data.py       # Synthetic SaaS dataset generator
│   └── saas_churn.csv         # Raw dataset (5,000 customers)
│
├── src/
│   ├── features/
│   │   └── build_features.py  # Feature engineering + encoding + scaling
│   └── models/
│       ├── train_model.py     # Model training, evaluation, MLflow logging
│       ├── best_model.pkl     # Saved best model (auto-generated)
│       └── encoders.pkl       # Saved encoders/scalers (auto-generated)
│
├── dags/
│   └── churn_pipeline_dag.py  # Airflow DAG + local pipeline runner
│
├── app/
│   └── app.py                 # Flask REST API (/predict endpoint)
│
├── mlruns/                    # MLflow experiment logs (auto-generated)
│   ├── experiment_1/          # Per-run params + metrics
│   └── summary.json           # Best model summary
│
├── notebooks/
│   └── EDA_and_Analysis.ipynb # Exploratory data analysis
│
├── requirements.txt
└── README.md
```

---

## 🧠 Key Technical Decisions

**Why Random Forest over Logistic Regression?**
Churn is driven by non-linear interactions (e.g. a customer on a Free plan who also hasn't logged in is far more at-risk than either factor alone). Tree-based models capture these interactions naturally.

**Why `class_weight="balanced"`?**
Only ~25% of customers churn, so the dataset is imbalanced. Without this, the model would predict "no churn" for everyone and still get 75% accuracy — which is useless. Balanced weighting forces the model to learn both classes equally.

**Why Recall over Precision as primary metric?**
Missing a churner (false negative) = lost revenue. Incorrectly flagging a healthy customer (false positive) = one unnecessary email. The business cost of a false negative is much higher.

---

## 🛠 Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.9+ |
| ML | scikit-learn (RF, GB, LR) |
| Orchestration | Apache Airflow (DAG) |
| Experiment Tracking | MLflow |
| API | Flask |
| Data | Pandas, NumPy |
| Cloud (production) | AWS S3 (data store), EC2 (model serving) |

---

## 👩‍💻 Author

**Priyanka S** — Junior Data Analyst  
📧 priyankas1031@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/PriyankaS1031) | [GitHub](https://github.com/priyankas286)

---

*Built as a production-style portfolio project demonstrating end-to-end ML pipeline engineering.*
