# Automated Pipeline for Intelligent Document Processing and Interactive Exploration with RAG

## Overview

This project automates the ingestion, processing, and retrieval of documents, providing users with interactive document exploration and question-answering capabilities using a Retrieval-Augmented Generation (RAG) model. It integrates tools such as Apache Airflow, FastAPI, and Streamlit for seamless automation, efficient processing, and interactive user experience.

## Attestation and Contribution Declaration

WE ATTEST THAT WE HAVEN'T USED ANY OTHER STUDENTS' WORK IN OUR ASSIGNMENT AND ABIDE BY THE POLICIES LISTED IN THE STUDENT HANDBOOK.

**Contribution Breakdown**:
- Chiu Meng Che: 34%
- Shraddha Bhandarkar: 33%
- Kefan Zhang: 33%

## Workflow Diagram

The workflow diagram outlines the integration of major components including Apache Airflow, Streamlit, and FastAPI. Documents are ingested and processed automatically, with user-friendly APIs and a frontend interface for querying and exploring the results.

## Key Features

### **Automated Document Processing**
- **Data Ingestion**: Apache Airflow automates the extraction and transformation of document data.
- **Document Parsing**: Extracted documents are converted into structured formats such as Markdown for further processing.

### **Backend API with FastAPI**
- **Document Exploration**: Provides REST API endpoints to explore documents and their content, including metadata and extracted text.
- **Embedding Integration**: Supports semantic search through document embeddings.

### **User Interface with Streamlit**
- **Document Interaction**: A clean, intuitive UI for users to query and explore documents. Users can view summaries and insights directly in the interface.
- **Customizable Reports**: Supports exporting results as PDFs or Codelabs for instructional use.

### **AI-Powered Insights**
- **RAG Integration**: Supports Retrieval-Augmented Generation for in-depth content analysis and dynamic answers to user queries.
- **Summarization**: AI-powered summaries help users quickly grasp document content.

## Project Structure

```bash
│  .env
│  README.md
│  
├── airflow
│   ├── dags
│   │   ├── __pycache__
│   │   └── modules
│   │       ├── __pycache__
│   │       ├── __init__.py
│   │       ├── data_processing.py
│   │       └── publications_dag.py
│   ├── logs
│   ├── airflow.cfg
│   ├── poetry.lock
│   └── pyproject.toml
│
├── backend
│   ├── __pycache__
│   ├── __init__.py
│   ├── agents.py
│   ├── arxiv_test.py
│   ├── test.py
│   ├── poetry.lock
│   └── pyproject.toml
│
├── frontend
│   ├── codelabs.md
│   ├── research_report.pdf
│   ├── streamlit_app.py
│   ├── poetry.lock
│   └── pyproject.toml

## Prerequisites

**Poetry**: A Python dependency management tool.
- Install Poetry by following the instructions: [Poetry Installation Guide](https://python-poetry.org/docs/#installation)
- Verify installation:
  ```bash
  poetry --version
  ```

**Python 3.9+**: The project requires Python 3.9 or above.
- Verify installation:
  ```bash
  python3 --version
  ```

Ensure all prerequisites are installed before proceeding to deployment.

## Installation and Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your_repository/ai-driven-document-system.git
   cd ai-driven-document-system
   ```

2. **Environment Setup**
   - Use Poetry to install all dependencies:
     ```bash
     poetry install
     ```

3. **Running the Application**
   - Start FastAPI:
     ```poetry run uvicorn backend.agents:app --reload
     ```
   - Start Streamlit：
      ```streamlit run frontend/streamlit_app.py
     ```  

4. **Access the Application**
   - Streamlit frontend is accessible at `http://localhost:8501`
   - FastAPI backend documentation (Swagger UI) is available at `http://localhost:8000/docs`




## Resources

- **Streamlit Documentation: https://docs.streamlit.io/**
- **FastAPI Documentation: https://fastapi.tiangolo.com/**
- **Airflow Documentation: https://airflow.apache.org/**

## Demonstration Video

[Click here to watch the video demonstration](https://youtu.be/MyrS6RYSmA4)

## Codelabs Documentation

[Click here to view the Codelabs documentation](https://codelabs-preview.appspot.com/?file_id=1gBQts95I9VOnikyCroLEi9CNN_CU3dvoj7Q-2rhd6xU/edit?tab=t.0#0)


