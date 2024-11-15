import streamlit as st
import os
import json
from research_agent import build_graph, tool_str_to_func, AgentState
from langgraph.graph import END
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from pinecone import Pinecone

# Set up environment variables (ensure your API keys are securely stored)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    st.error("Please set your Pinecone API key and index name.")
    st.stop()

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Function to get the list of publications from Pinecone
def get_publications():
    # Note: Pinecone does not support listing all unique metadata values directly.
    # For this example, we will retrieve a sample of vectors to extract publications.
    # In a production environment, maintain a separate list of publications.
    query_result = index.query(vector=[0]*384, top_k=1000, include_metadata=True)
    publications = set()
    for match in query_result['matches']:
        metadata = match.get('metadata', {})
        publication = metadata.get('publication')
        if publication:
            publications.add(publication)
    return sorted(publications)

publications = get_publications()

st.title("Research Agent Interface")

# Document selection
selected_publication = st.selectbox("Select a document for research:", publications)

# Initialize session state for the research session
if 'research_session' not in st.session_state:
    st.session_state['research_session'] = []

# User input
user_question = st.text_input("Enter your research question:")

if st.button("Ask"):
    if user_question:
        # Build the agent state
        state = AgentState(
            input=user_question,
            chat_history=[],
            intermediate_steps=[],
            publication=selected_publication  # Include the selected publication
        )

        # Build the graph and runnable
        graph, runnable = build_graph()

        # Run the agent
        result = runnable.invoke(state)

        # Extract the final output from the result
        if result['intermediate_steps']:
            final_action = result['intermediate_steps'][-1]
            if final_action.tool == 'final_answer':
                output = final_action.log  # This should be the final report dict
                # Since the output is a string, we need to parse it
                if isinstance(output, str):
                    try:
                        output_dict = json.loads(output.replace("'", '"'))
                    except json.JSONDecodeError:
                        st.error("Failed to parse the final answer output.")
                        output_dict = {}
                else:
                    output_dict = output
                if output_dict:
                    report = build_report(output_dict)
                    # Store the question and answer
                    st.session_state['research_session'].append({
                        'question': user_question,
                        'answer': report
                    })
                    st.write("### Answer:")
                    st.write(report)
                else:
                    st.error("No output from final answer.")
            else:
                st.error("The agent did not produce a final answer.")
        else:
            st.error("No intermediate steps found in the result.")
    else:
        st.error("Please enter a question.")

# Display previous questions and answers
if st.session_state['research_session']:
    st.write("## Research Session History:")
    for idx, qa in enumerate(st.session_state['research_session']):
        st.write(f"**Question {idx+1}:** {qa['question']}")
        st.write(f"**Answer:**")
        st.write(qa['answer'])
        st.write("---")

# Option to export results
if st.button("Export Results"):
    if st.session_state['research_session']:
        # Build the report using the build_report function
        # Collect all outputs into one report
        output = {
            "introduction": f"Research report on '{selected_publication}'.",
            "research_steps": [qa['question'] for qa in st.session_state['research_session']],
            "main_body": "\n\n".join([qa['answer'] for qa in st.session_state['research_session']]),
            "conclusion": "This concludes our research findings.",
            "sources": []
        }
        report = build_report(output)
        # Generate PDF (optional: requires additional libraries like FPDF or ReportLab)
        # For simplicity, we'll allow the user to download the report as a text file
        st.download_button(
            label="Download Report",
            data=report,
            file_name="research_report.txt",
            mime="text/plain"
        )
    else:
        st.error("No research session data to export.")

# Function to format the report output
def build_report(output: dict):
    research_steps = output["research_steps"]
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output["sources"]
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])
    return f"""
INTRODUCTION
------------
{output["introduction"]}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output["main_body"]}

CONCLUSION
----------
{output["conclusion"]}

SOURCES
-------
{sources}
"""
