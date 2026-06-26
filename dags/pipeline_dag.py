from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'loan_default_pipeline',
    default_args=default_args,
    description='Monthly training + inference pipeline (2024-09 onwards)',
    schedule_interval='0 0 1 * *',
    start_date=datetime(2024, 9, 1),
    end_date=datetime(2025, 7, 1), # ends July because last features' snapshot are 2025-01-01
    catchup=True,
    max_active_runs=1,
) as dag:

    # -------------------------------------------------------------------------
    # LABEL STORE
    # -------------------------------------------------------------------------

    dep_check_lms = BashOperator(
        task_id='dep_check_lms',
        bash_command='cd /opt/airflow && python3 utils/dep_check/dep_check_csv_lms.py',
    )

    bronze_lms = BashOperator(
        task_id='bronze_lms',
        bash_command='cd /opt/airflow && python3 utils/medallion/bronze/bronze_lms.py --snapshotdate "{{ ds }}"',
    )

    silver_lms = BashOperator(
        task_id='silver_lms',
        bash_command='cd /opt/airflow && python3 utils/medallion/silver/silver_lms.py --snapshotdate "{{ ds }}"',
    )

    gold_label_store = BashOperator(
        task_id='gold_label_store',
        bash_command='cd /opt/airflow && python3 utils/medallion/gold/gold_label.py --snapshotdate "{{ ds }}"',
    )

    label_store_completed = DummyOperator(task_id='label_store_completed')

    dep_check_lms >> bronze_lms >> silver_lms >> gold_label_store >> label_store_completed


    # -------------------------------------------------------------------------
    # FEATURE STORE
    # -------------------------------------------------------------------------

    dep_check_fin = BashOperator(
        task_id='dep_check_fin',
        bash_command='cd /opt/airflow && python3 utils/dep_check/dep_check_csv_fin.py',
    )

    dep_check_attr = BashOperator(
        task_id='dep_check_attr',
        bash_command='cd /opt/airflow && python3 utils/dep_check/dep_check_csv_attr.py',
    )

    dep_check_cs = BashOperator(
        task_id='dep_check_cs',
        bash_command='cd /opt/airflow && python3 utils/dep_check/dep_check_csv_cs.py',
    )

    bronze_financials = BashOperator(
        task_id='bronze_financials',
        bash_command='cd /opt/airflow && python3 utils/medallion/bronze/bronze_financials.py --snapshotdate "{{ ds }}"',
    )

    bronze_attributes = BashOperator(
        task_id='bronze_attributes',
        bash_command='cd /opt/airflow && python3 utils/medallion/bronze/bronze_attributes.py --snapshotdate "{{ ds }}"',
    )

    bronze_clickstream = BashOperator(
        task_id='bronze_clickstream',
        bash_command='cd /opt/airflow && python3 utils/medallion/bronze/bronze_clickstream.py --snapshotdate "{{ ds }}"',
    )

    silver_fin_attr = BashOperator(
        task_id='silver_fin_attr',
        bash_command='cd /opt/airflow && python3 utils/medallion/silver/silver_fin_attr.py --snapshotdate "{{ ds }}"',
    )

    silver_clickstream = BashOperator(
        task_id='silver_clickstream',
        bash_command='cd /opt/airflow && python3 utils/medallion/silver/silver_clickstream.py --snapshotdate "{{ ds }}"',
    )

    gold_features = BashOperator(
        task_id='gold_features',
        bash_command='cd /opt/airflow && python3 utils/medallion/gold/gold_features.py --snapshotdate "{{ ds }}"',
    )

    feature_store_completed = DummyOperator(task_id='feature_store_completed')

    dep_check_fin >> bronze_financials >> silver_fin_attr >> gold_features
    dep_check_attr >> bronze_attributes >> silver_fin_attr >> gold_features
    dep_check_cs >> bronze_clickstream >> silver_clickstream >> gold_features
    gold_features >> feature_store_completed


    # -------------------------------------------------------------------------
    # MODEL TRAINING
    # 2024-09-01: trains LR + XGB, selects champion by Gini
    # 2024-10-01+: AutoML retrains on accumulated data
    # -------------------------------------------------------------------------

    model_train_and_automl = BashOperator(
        task_id='model_train_and_automl',
        bash_command='cd /opt/airflow && python3 utils/ml/train_and_automl.py --snapshotdate "{{ ds }}"',
    )

    label_store_completed >> model_train_and_automl
    feature_store_completed >> model_train_and_automl


    # -------------------------------------------------------------------------
    # MODEL INFERENCE
    # Skips 2024-09-01 (no champion yet), starts from 2024-10-01
    # -------------------------------------------------------------------------

    model_inference = BashOperator(
        task_id='model_inference',
        bash_command='cd /opt/airflow && python3 utils/ml/inference.py --snapshotdate "{{ ds }}"',
    )

    label_store_completed >> model_inference
    feature_store_completed >> model_inference


    # -------------------------------------------------------------------------
    # MODEL MONITORING
    # -------------------------------------------------------------------------

    model_monitor = BashOperator(
        task_id='model_monitor',
        bash_command='cd /opt/airflow && python3 utils/monitoring/monitoring.py --snapshotdate "{{ ds }}"',
    )

    model_inference >> model_monitor
