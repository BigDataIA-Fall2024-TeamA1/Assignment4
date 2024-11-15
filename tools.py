import os
import re
from typing import List
from getpass import getpass
import requests
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from serpapi import GoogleSearch
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI

# Load environment variables
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify that all required environment variables are set
if not api_key or not index_name or not openai_api_key:
    raise ValueError("PINECONE_API_KEY, PINECONE_INDEX_NAME, and OPENAI_API_KEY must be set in the environment.")

# Initialize encoder
encoder = SentenceTransformer('all-MiniLM-L6-v2')
dims = encoder.get_sentence_embedding_dimension()

# Configure Pinecone client and check if the specified index exists
try:
    pc = Pinecone(api_key=api_key)
    available_indexes = [idx['name'] for idx in pc.list_indexes().get('indexes', [])]
    print("Available indexes:", available_indexes)

    if index_name not in available_indexes:
        raise ValueError(f"Index '{index_name}' does not exist in Pinecone. Please check the index name.")
    else:
        index = pc.Index(index_name)
        print(f"Successfully connected to Pinecone index '{index_name}'.")

    # Retrieve and print index stats
    try:
        index_stats = index.describe_index_stats()
        print("Index stats:", index_stats)
    except Exception as e:
        print("Failed to retrieve index stats:", e)
        raise e

except Exception as e:
    print("Error initializing Pinecone or loading index:", e)
    raise e

# Regex pattern for extracting abstracts if needed
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

# Define tools
@tool("web_search")
def web_search(query: str):
    """Searches the web using Google Search API."""
    serpapi_key = os.getenv("SERPAPI_KEY") or getpass("SerpAPI key: ")
    search = GoogleSearch({"engine": "google", "api_key": serpapi_key, "q": query, "num": 5})
    results = search.get_dict().get("organic_results", [])
    return "\n---\n".join(["\n".join([x.get("title", ""), x.get("snippet", ""), x.get("link", "")]) for x in results])

def format_rag_contexts(matches: list) -> str:
    contexts = [
        f"Publication: {x['metadata'].get('publication', '')}\n"
        f"Source: {x['metadata'].get('source', '')}\n"
        f"Text: {x['metadata'].get('text', '')}"
        for x in matches
    ]
    return "\n---\n".join(contexts)

@tool("rag_search")
def rag_search(query: str) -> str:
    """Searches the Pinecone index using a query."""
    query_vector = encoder.encode([query])
    try:
        search_result = index.query(vector=query_vector[0], top_k=5, include_metadata=True)
        return format_rag_contexts(search_result["matches"])
    except Exception as e:
        print("Error performing rag_search:", e)
        return "An error occurred during rag_search."

@tool("rag_search_filter")
def rag_search_filter(query: str, publication: str) -> str:
    """Searches the Pinecone index with a filter for a specific publication."""
    query_vector = encoder.encode([query])
    try:
        search_result = index.query(vector=query_vector[0], top_k=6, include_metadata=True, filter={"publication": publication})
        return format_rag_contexts(search_result["matches"])
    except Exception as e:
        print("Error performing rag_search_filter:", e)
        return "An error occurred during rag_search_filter."

@tool("final_answer")
def final_answer(introduction: str, research_steps: str, main_body: str, conclusion: str, sources: str) -> str:
    """Generates a structured research report."""
    research_steps_str = "\n".join([f"- {step}" for step in research_steps]) if isinstance(research_steps, list) else research_steps
    sources_str = "\n".join([f"- {source}" for source in sources]) if isinstance(sources, list) else sources
    return (
        f"Introduction:\n{introduction}\n\n"
        f"Research Steps:\n{research_steps_str}\n\n"
        f"Main Body:\n{main_body}\n\n"
        f"Conclusion:\n{conclusion}\n\n"
        f"Sources:\n{sources_str}"
    )

# System prompt for ChatPromptTemplate
system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query, you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (i.e., if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad), use the final_answer
tool."""

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

# Initialize language model
llm = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key, temperature=0)

# List of tools
tools = [rag_search_filter, rag_search, web_search, final_answer]

def create_scratchpad(intermediate_steps: List[AgentAction]) -> str:
    return "\n---\n".join(
        f"Tool: {action.tool}, input: {action.tool_input}\nOutput: {action.log}"
        for action in intermediate_steps if action.log != "TBD"
    )

# Define oracle
oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(x["intermediate_steps"]),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

# Define function to run oracle
def run_oracle(state: dict):
    print("run_oracle")
    print(f"intermediate_steps: {state['intermediate_steps']}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")
    return {"intermediate_steps": [action_out]}
