"""
app/app.py
-----------
Flask web application that serves the trained churn prediction model.

ENDPOINTS:
  GET  /           → Dashboard homepage
  POST /predict    → JSON API: send customer data, get churn prediction
  GET  /health     → Health check (useful for cloud deployment)

HOW IT WORKS:
  1. On startup, loads the saved model from src/models/best_model.pkl
  2. /predict receives customer data as JSON
  3. Runs feature engineering (same steps as training)
  4. Returns: churn probability + risk level + business recommendation
"""

import os, sys, pickle, json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = Flask(__name__)

# ── Load model on startup ──────────────────────────────────
MODEL_PATH    = "src/models/best_model.pkl"
ENCODERS_PATH = "src/models/encoders.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    return bundle, encoders

try:
    MODEL_BUNDLE, ENCODERS = load_model()
    print(f"✅ Model loaded: {MODEL_BUNDLE['model_name']}")
    print(f"   Trained at : {MODEL_BUNDLE['trained_at']}")
    print(f"   ROC-AUC    : {MODEL_BUNDLE['metrics']['roc_auc']}")
except Exception as e:
    print(f"⚠️  Model not found. Run pipeline first: python dags/churn_pipeline_dag.py")
    MODEL_BUNDLE, ENCODERS = None, None


def preprocess_input(data: dict) -> pd.DataFrame:
    """Apply the same feature engineering used during training."""
    df = pd.DataFrame([data])

    # ── Engineered features (must match build_features.py exactly) ──
    df["revenue_per_user"]  = df["monthly_charges"] / (df["num_users"] + 1)
    df["support_intensity"] = df["num_support_tickets"] / (df["tenure_months"] + 1)

    login_score    = 1 - (df["last_login_days_ago"] / 90)
    login_score    = login_score.clip(0, 1)
    adoption_score = df["feature_adoption_pct"] / 100
    df["engagement_score"] = ((login_score * 0.5) + (adoption_score * 0.5)) * 100

    df["ltv_proxy"]    = df["total_charges"] / (df["tenure_months"] + 1)
    df["at_risk_flag"] = ((df["last_login_days_ago"] > 30) &
                          (df["feature_adoption_pct"] < 30)).astype(int)

    # ── Encode categoricals ──
    for col in ["plan_type", "payment_method"]:
        df[col] = ENCODERS[col].transform(df[col])

    # ── Scale numerics ──
    df[ENCODERS["num_cols"]] = ENCODERS["scaler"].transform(df[ENCODERS["num_cols"]])

    return df[MODEL_BUNDLE["features"]]


def get_recommendation(churn_prob: float, data: dict) -> str:
    """Generate a business recommendation based on churn risk."""
    if churn_prob >= 0.75:
        return (f"🚨 HIGH RISK: Assign a Customer Success Manager immediately. "
                f"Offer a {int(churn_prob*30)}% discount on renewal. "
                f"Schedule a product review call within 48 hours.")
    elif churn_prob >= 0.45:
        return (f"⚠️ MEDIUM RISK: Trigger an automated re-engagement email sequence. "
                f"Highlight unused features. Consider an upgrade offer.")
    else:
        return (f"✅ LOW RISK: Customer is healthy. "
                f"Good candidate for upsell to a higher plan.")


# ── Routes ────────────────────────────────────────────────

@app.route("/health")
def health():
    """Health check endpoint — used by cloud load balancers."""
    status = "ok" if MODEL_BUNDLE else "model_not_loaded"
    return jsonify({"status": status, "model": MODEL_BUNDLE["model_name"] if MODEL_BUNDLE else None})


@app.route("/", methods=["GET"])
def index():
    """Simple dashboard showing model info and how to use the API."""
    metrics = MODEL_BUNDLE["metrics"] if MODEL_BUNDLE else {}
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SaaS Churn Predictor</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; max-width: 860px;
                    margin: 40px auto; padding: 20px; background: #0f172a; color: #e2e8f0; }}
            h1   {{ color: #38bdf8; font-size: 2rem; margin-bottom: 4px; }}
            .sub {{ color: #94a3b8; margin-bottom: 32px; }}
            .card {{ background: #1e293b; border-radius: 12px; padding: 24px;
                     margin-bottom: 20px; border: 1px solid #334155; }}
            .metric {{ display: inline-block; margin-right: 24px; }}
            .metric span {{ display: block; color: #38bdf8; font-size: 1.6rem;
                             font-weight: 700; }}
            .metric label {{ color: #64748b; font-size: 0.8rem; text-transform: uppercase; }}
            code {{ background: #0f172a; padding: 16px; border-radius: 8px;
                    display: block; font-size: 0.82rem; color: #a5f3fc;
                    border: 1px solid #334155; white-space: pre; overflow-x: auto; }}
            h2 {{ color: #38bdf8; font-size: 1.1rem; margin-bottom: 12px; }}
        </style>
    </head>
    <body>
        <h1>🔮 SaaS Churn Predictor</h1>
        <p class="sub">Production ML API · Built by Priyanka S · Random Forest + Feature Engineering</p>

        <div class="card">
            <h2>📊 Model Performance</h2>
            <div class="metric"><span>{metrics.get('roc_auc','—')}</span><label>ROC-AUC</label></div>
            <div class="metric"><span>{metrics.get('f1_score','—')}</span><label>F1 Score</label></div>
            <div class="metric"><span>{metrics.get('recall','—')}</span><label>Recall</label></div>
            <div class="metric"><span>{metrics.get('accuracy','—')}</span><label>Accuracy</label></div>
        </div>

        <div class="card">
            <h2>🔌 API Usage — POST /predict</h2>
<code>curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{{
    "tenure_months": 3,
    "monthly_charges": 1200,
    "total_charges": 3600,
    "num_users": 2,
    "num_support_tickets": 6,
    "last_login_days_ago": 45,
    "feature_adoption_pct": 15,
    "plan_type": "Free",
    "payment_method": "UPI"
  }}'</code>
        </div>
    </body>
    </html>
    """
    return html


@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint.
    Receives customer data → returns churn probability + recommendation.
    """
    if not MODEL_BUNDLE:
        return jsonify({"error": "Model not loaded. Run pipeline first."}), 503

    try:
        data = request.get_json(force=True)

        # Validate required fields
        required = ["tenure_months", "monthly_charges", "total_charges",
                    "num_users", "num_support_tickets", "last_login_days_ago",
                    "feature_adoption_pct", "plan_type", "payment_method"]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Preprocess and predict
        X           = preprocess_input(data)
        churn_prob  = float(MODEL_BUNDLE["model"].predict_proba(X)[0][1])
        prediction  = int(churn_prob >= 0.5)

        # Risk level
        if churn_prob >= 0.75:
            risk_level = "HIGH"
        elif churn_prob >= 0.45:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        response = {
            "churn_probability" : round(churn_prob, 4),
            "churn_prediction"  : prediction,
            "risk_level"        : risk_level,
            "recommendation"    : get_recommendation(churn_prob, data),
            "model_used"        : MODEL_BUNDLE["model_name"],
            "model_roc_auc"     : MODEL_BUNDLE["metrics"]["roc_auc"]
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n🌐 Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000)
