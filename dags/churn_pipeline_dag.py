"""
dags/churn_pipeline_dag.py
---------------------------
This is the Airflow DAG (Directed Acyclic Graph) that
orchestrates the ENTIRE churn prediction pipeline.

Think of a DAG as a recipe with steps in a specific order:
  Step 1 → Step 2 → Step 3 (each step must finish before the next begins)

In production (real Airflow), you would deploy this file to
your Airflow /dags folder and it runs on a schedule (e.g. every week).

For this project, we also include a run_pipeline() function
that simulates the same flow locally — run this file directly
to execute the full pipeline.

PIPELINE STEPS:
  [ingest_data] → [build_features] → [train_model] → [evaluate_model] → [serve_ready]
"""

import os, sys, json
from datetime import datetime, timedelta

# ── Make imports work from project root ─────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ───────────────────────────────────────────────────────────
# SECTION A: Airflow DAG definition
# (Only runs when Airflow is installed — safe to keep here)
# ───────────────────────────────────────────────────────────

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator

    # Default arguments applied to every task in this DAG
    default_args = {
        "owner"           : "priyanka_s",
        "depends_on_past" : False,
        "email_on_failure": False,
        "retries"         : 1,
        "retry_delay"     : timedelta(minutes=5),
    }

    # Define the DAG — runs every Sunday at midnight
    dag = DAG(
        dag_id            = "saas_churn_prediction_pipeline",
        default_args      = default_args,
        description       = "Weekly SaaS customer churn prediction pipeline",
        schedule_interval = "@weekly",
        start_date        = datetime(2024, 1, 1),
        catchup           = False,
        tags              = ["churn", "ml", "saas"],
    )

    # Task 1: Data ingestion
    t1 = PythonOperator(
        task_id         = "ingest_data",
        python_callable = lambda: __import__("data.generate_data"),
        dag             = dag,
    )

    # Task 2: Feature engineering
    t2 = PythonOperator(
        task_id         = "build_features",
        python_callable = lambda: __import__("src.features.build_features",
                            fromlist=["build_features"]).build_features(
                            "data/saas_churn.csv"),
        dag             = dag,
    )

    # Task 3: Model training
    t3 = PythonOperator(
        task_id         = "train_model",
        python_callable = lambda: __import__("src.models.train_model",
                            fromlist=["train"]).train(),
        dag             = dag,
    )

    # Task 4: Notify when done
    t4 = PythonOperator(
        task_id         = "pipeline_complete",
        python_callable = lambda: print("✅ Churn pipeline complete. Model ready for serving."),
        dag             = dag,
    )

    # Set the ORDER of execution: t1 → t2 → t3 → t4
    t1 >> t2 >> t3 >> t4

    print("Airflow DAG registered: saas_churn_prediction_pipeline")

except ImportError:
    # Airflow not installed — that's fine, use run_pipeline() below
    pass


# ───────────────────────────────────────────────────────────
# SECTION B: Local pipeline runner (no Airflow needed)
# Run: python dags/churn_pipeline_dag.py
# ───────────────────────────────────────────────────────────

def task_ingest_data():
    """Task 1: Generate / refresh the raw dataset."""
    print("\n" + "="*55)
    print("  TASK 1 / 4  →  Data Ingestion")
    print("="*55)
    # In a real project this pulls from AWS S3, a database, or API
    # Here we (re)generate our synthetic dataset
    import subprocess
    result = subprocess.run(["python", "data/generate_data.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Data ingestion failed:\n{result.stderr}")
    print("  ✅ Task 1 complete")


def task_build_features():
    """Task 2: Feature engineering."""
    print("\n" + "="*55)
    print("  TASK 2 / 4  →  Feature Engineering")
    print("="*55)
    from src.features.build_features import build_features
    build_features("data/saas_churn.csv", "data/features.csv")
    print("  ✅ Task 2 complete")


def task_train_model():
    """Task 3: Model training & experiment tracking."""
    print("\n" + "="*55)
    print("  TASK 3 / 4  →  Model Training + MLflow Logging")
    print("="*55)
    from src.models.train_model import train
    model, metrics = train()
    print(f"  ✅ Task 3 complete  |  Best ROC-AUC: {metrics['roc_auc']}")
    return metrics


def task_pipeline_complete(metrics: dict):
    """Task 4: Log final status."""
    print("\n" + "="*55)
    print("  TASK 4 / 4  →  Pipeline Complete")
    print("="*55)
    status = {
        "status"      : "SUCCESS",
        "completed_at": datetime.now().isoformat(),
        "model_metrics": metrics
    }
    with open("mlruns/last_run_status.json", "w") as f:
        json.dump(status, f, indent=2)
    print(f"  📋 Status: {status['status']}")
    print(f"  🕐 Completed at: {status['completed_at']}")
    print(f"  🎯 Final ROC-AUC: {metrics['roc_auc']}")
    print("\n  ✅ Model is ready. Launch Flask app: python app/app.py")
    print("="*55)


def run_pipeline():
    """
    Runs the full pipeline locally, mimicking Airflow task execution.
    This is what you demo in interviews / README.
    """
    print("\n🚀 Starting SaaS Churn Prediction Pipeline")
    print(f"   Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    start = datetime.now()

    task_ingest_data()
    task_build_features()
    metrics = task_train_model()
    task_pipeline_complete(metrics)

    elapsed = (datetime.now() - start).seconds
    print(f"\n⏱  Total pipeline time: {elapsed}s")


if __name__ == "__main__":
    run_pipeline()
