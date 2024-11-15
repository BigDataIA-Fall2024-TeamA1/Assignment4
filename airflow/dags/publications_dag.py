# publications_data.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from modules.data_processing import (
    download_publications,
    parsing_publications,
    embedding_and_upload_pinecone,
    delete_local_publications
)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 10, 1),
    'retries': 1,
}

with DAG(
    'publications_processing_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False,
) as dag:

    task_download = PythonOperator(
        task_id='download_publications',
        python_callable=download_publications,
    )

    task_parse = PythonOperator(
        task_id='parsing_publications',
        python_callable=parsing_publications,
    )

    task_embed_upload = PythonOperator(
        task_id='embedding_and_upload_pinecone',
        python_callable=embedding_and_upload_pinecone,
    )

    task_cleanup = PythonOperator(
        task_id='delete_local_publications',
        python_callable=delete_local_publications,
    )

    task_download >> task_parse >> task_embed_upload >> task_cleanup
