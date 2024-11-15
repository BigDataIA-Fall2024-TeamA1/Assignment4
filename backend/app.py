# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from agents import ask_question
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# 允许 CORS（跨域资源共享）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 为了开发方便，允许所有来源，生产环境中请设置为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AIRFLOW_API_URL = os.environ.get('AIRFLOW_API_URL', 'http://localhost:8080/api/v1')

class QuestionRequest(BaseModel):
    question: str
    document: str

@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    # 将选定的文档信息添加到问题中
    question_with_context = f"Document: {request.document}\nQuestion: {request.question}"
    response = ask_question(question_with_context)
    return {"response": response}

class PipelineRequest(BaseModel):
    dag_id: str

@app.post("/trigger_pipeline")
def trigger_pipeline(request: PipelineRequest):
    # 触发 Airflow DAG
    dag_id = request.dag_id
    trigger_url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns"
    response = requests.post(trigger_url, json={})
    if response.status_code == 200 or response.status_code == 409:
        return {"message": f"DAG {dag_id} triggered successfully.", "dag_run_id": response.json().get('dag_run_id')}
    else:
        return {"error": f"Failed to trigger DAG {dag_id}.", "details": response.text}

@app.get("/dag_status/{dag_id}")
def get_dag_status(dag_id: str):
    # 获取最新的 DAG 运行状态
    url = f"{AIRFLOW_API_URL}/dags/{dag_id}/dagRuns?order_by=-execution_date&limit=1"
    response = requests.get(url)
    if response.status_code == 200:
        dag_runs = response.json().get('dag_runs', [])
        if dag_runs:
            dag_run = dag_runs[0]
            return {
                "dag_id": dag_id,
                "dag_run_id": dag_run['dag_run_id'],
                "state": dag_run['state'],
                "execution_date": dag_run['execution_date']
            }
        else:
            return {"message": f"No runs found for DAG {dag_id}."}
    else:
        return {"error": f"Failed to get status for DAG {dag_id}.", "details": response.text}
