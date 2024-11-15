# streamlit_app.py

import streamlit as st
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
import requests
import re
import subprocess

API_BASE_URL = "http://localhost:8000/publications"

# Initialize session state for saving results
if "session_results" not in st.session_state:
    st.session_state["session_results"] = []
if "displayed_result" not in st.session_state:
    st.session_state["displayed_result"] = None
if "pdf_ready" not in st.session_state:
    st.session_state["pdf_ready"] = False
if "codelabs_ready" not in st.session_state:
    st.session_state["codelabs_ready"] = False

def main():
    # Sidebar for document selection and save options
    st.sidebar.title("Document Selection")

    # Fetch publications from FastAPI service
    try:
        response = requests.get(API_BASE_URL)
        if response.status_code == 200:
            publications = response.json().get("publications", [])
        else:
            st.sidebar.error(f"Failed to retrieve publications: {response.status_code} - {response.text}")
            return
    except Exception as e:
        st.sidebar.error(f"An error occurred while fetching publications: {str(e)}")
        return

    if not publications:
        st.sidebar.warning("No publications found.")
        return

    # Document selection dropdown in sidebar
    selected_publication = st.sidebar.selectbox("Select a publication", publications)

    # Save Results options in sidebar
    st.sidebar.subheader("Save All Results")
    if st.sidebar.button("Save as PDF"):
        generate_pdf(st.session_state["session_results"])
        st.session_state["pdf_ready"] = True
    if st.session_state["pdf_ready"]:
        with open("research_report.pdf", "rb") as pdf_file:
            st.sidebar.download_button("Download PDF", data=pdf_file, file_name="research_report.pdf")
            st.session_state["pdf_ready"] = False  # Reset after download

    if st.sidebar.button("Save as Codelabs"):
        generate_codelabs(st.session_state["session_results"])
        st.session_state["codelabs_ready"] = True
    if st.session_state["codelabs_ready"]:
        with open("codelabs_output.zip", "rb") as zip_file:
            st.sidebar.download_button("Download Codelabs", data=zip_file, file_name="codelabs.zip")
            st.session_state["codelabs_ready"] = False  # Reset after download

    # Main UI on the right side
    st.title("Publication Query Application")

    # Query input and buttons for agent selection
    query = st.text_input("Enter your query here")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Use RAG Search Filter"):
            handle_query(query, selected_publication, "rag_search_filter")
    with col2:
        if st.button("Use Web Search"):
            handle_query(query, selected_publication, "web_search")

    # Display the result in a large area occupying the right side
    if st.session_state["displayed_result"]:
        display_result(st.session_state["displayed_result"])

# Function to handle query execution based on selected agent
def handle_query(query, publication, agent):
    if not query.strip():
        st.error("Please enter a query.")
        return
    
    # Prepare data for request
    data = {
        "input": query,
        "publication": publication if agent == "rag_search_filter" else None
    }
    try:
        # Main query request to FastAPI
        response = requests.post("http://localhost:8000/query", json=data)
        if response.status_code == 200:
            report = response.json()
            # Fetch ArXiv papers related to the query
            arxiv_papers = fetch_arxiv_papers(query)
            report["arxiv_results"] = [
                f"Title: {title}\nLink: {link}" for title, _, link in arxiv_papers
            ]
            st.session_state["displayed_result"] = report
            st.session_state["session_results"].append(report)
        else:
            st.error(f"Request failed: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Fetch relevant ArXiv papers based on a query
def fetch_arxiv_papers(query: str, max_results: int = 5):
    """
    Fetch relevant ArXiv papers based on a query.
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        return []
    
    # Parse the response (Atom XML format)
    papers = []
    entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
    
    for entry in entries:
        title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL).group(1).strip()
        link = re.search(r'<id>(.*?)</id>', entry, re.DOTALL).group(1).strip()
        papers.append((title, "", link))
    
    return papers

# Function to display results in a structured format
def display_result(report):
    st.subheader("Introduction")
    st.write(report.get("introduction", ""))
    st.subheader("Research Steps")
    st.write(report.get("research_steps", ""))
    st.subheader("Main Body")
    st.write(report.get("main_body", ""))
    st.subheader("Conclusion")
    st.write(report.get("conclusion", ""))

    # Display Sources as Hyperlinks
    st.subheader("Sources")
    sources = report.get("sources", "")
    if sources:
        source_lines = sources.split("\n")
        for index, source in enumerate(source_lines, start=1):
            # Check if source includes a URL, else display as plain text
            if "http" in source:
                title, url = source.rsplit(" ", 1)  # Separate title from URL
                st.markdown(f"{index}. [{title}]({url})")
            else:
                st.write(f"{index}. {source}")
    
    # Display Related ArXiv Papers with only title and link
    if report.get("arxiv_results"):
        st.subheader("Related ArXiv Papers")
        for arxiv_paper in report["arxiv_results"]:
            title, link = arxiv_paper.split("\nLink: ")
            st.markdown(f"- [{title}]({link})")  # Display as a link in a bullet list

# Generate PDF using ReportLab
def generate_pdf(results):
    if not results:
        st.sidebar.error("No results to save.")
        return
    pdf_path = "research_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    styles = getSampleStyleSheet()
    elements = []

    for idx, res in enumerate(results, start=1):
        elements.append(Paragraph(f"### Result {idx}", styles["Heading1"]))
        elements.append(Paragraph("Introduction", styles["Heading2"]))
        elements.append(Paragraph(res.get("introduction", ""), styles["BodyText"]))
        elements.append(Paragraph("Research Steps", styles["Heading2"]))
        elements.append(Paragraph(res.get("research_steps", ""), styles["BodyText"]))
        elements.append(Paragraph("Main Body", styles["Heading2"]))
        elements.append(Paragraph(res.get("main_body", ""), styles["BodyText"]))
        elements.append(Paragraph("Conclusion", styles["Heading2"]))
        elements.append(Paragraph(res.get("conclusion", ""), styles["BodyText"]))
        elements.append(Paragraph("Sources", styles["Heading2"]))
        elements.append(Paragraph(res.get("sources", ""), styles["BodyText"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)

# Generate Codelabs using claat
def generate_codelabs(results):
    if not results:
        st.sidebar.error("No results to save.")
        return
    # Create a temporary markdown file for claat
    markdown_path = "codelabs.md"
    with open(markdown_path, "w") as f:
        for idx, res in enumerate(results, start=1):
            f.write(f"# Result {idx}\n\n")
            f.write(f"## Introduction\n\n{res.get('introduction', '')}\n\n")
            f.write(f"## Research Steps\n\n{res.get('research_steps', '')}\n\n")
            f.write(f"## Main Body\n\n{res.get('main_body', '')}\n\n")
            f.write(f"## Conclusion\n\n{res.get('conclusion', '')}\n\n")
            f.write(f"## Sources\n\n{res.get('sources', '')}\n\n")
            f.write("\n---\n")

    # Run claat command to convert markdown to Codelabs format
    claat_output_dir = "codelabs_output"
    os.makedirs(claat_output_dir, exist_ok=True)
    subprocess.run(["claat", "export", markdown_path, "-o", claat_output_dir])

    # Zip the Codelabs output directory
    subprocess.run(["zip", "-r", "codelabs_output.zip", claat_output_dir])

if __name__ == "__main__":
    main()