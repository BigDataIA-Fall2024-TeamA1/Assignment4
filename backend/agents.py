import os
import time
import operator
import requests
import re
import xml.etree.ElementTree as ET
from getpass import getpass
from typing import TypedDict, Annotated
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from serpapi.google_search import GoogleSearch
from langgraph.graph import StateGraph, END
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

app = FastAPI()

# 初始化 OpenAI 编码器
encoder = OpenAIEncoder(name="text-embedding-3-small")

# 初始化 Pinecone
api_key = os.getenv("PINECONE_API_KEY") or getpass("Pinecone API key: ")
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
dims = len(encoder(["some random text"])[0])
index_name = "publications-vectors"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=dims,
        metric='dotproduct',
        spec=spec
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)
index.describe_index_stats()

# 定义工具
serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERPAPI_KEY") or getpass("SerpAPI key: ")
}

@tool("web_search")
def web_search(query: str):
    """使用 Google 搜索获取一般知识信息。"""
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 5
    })
    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x.get("snippet", ""), x["link"]]) for x in results]
    )
    return contexts

def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        text = (
            f"Publication: {x['metadata'].get('publication', 'N/A')}\n"
            f"Source File: {x['metadata'].get('source', 'N/A')}\n"
            f"Extracted Text: {x['metadata'].get('text', 'N/A')}\n"
        )
        contexts.append(text)
    context_str = "\n---\n".join(contexts)
    return context_str

@tool("rag_search_filter")
def rag_search_filter(query: str, publication: str):
    """Search for relevant information in Pinecone with filtering by publication."""
    xq = encoder([query])  # Generate vector embedding for the query
    print(f"Query vector: {xq}")  # Debug log

    # Perform similarity search in Pinecone
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"publication": publication})
    print(f"Query results: {xc}")  # Debug log

    # Check if matches are found
    if not xc["matches"]:
        return "No relevant documents found in Pinecone."

    # Format results into a readable context
    context_str = format_rag_contexts(xc["matches"])
    return context_str

# @tool("rag_search")
# def rag_search(query: str):
#     """使用自然语言查询查找 AI 方面的专业信息。"""
#     xq = encoder([query])
#     xc = index.query(vector=xq, top_k=2, include_metadata=True)
#     context_str = format_rag_contexts(xc["matches"])
#     return context_str

abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

import xml.etree.ElementTree as ET

def fetch_arxiv_papers(query: str, max_results: int = 5):
    """
    Fetch relevant ArXiv papers based on a query and extracts their abstracts.
    Args:
        query (str): Search query.
        max_results (int): Maximum number of papers to fetch.
    Returns:
        List of tuples (paper title, abstract, link).
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
        return f"Error fetching papers: {response.status_code}"
    
    # Parse the response XML
    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        link = entry.find("{http://www.w3.org/2005/Atom}id").text.strip()
        papers.append((title, summary, link))
    
    if not papers:
        return "No ArXiv papers found for this query."

    return papers

@tool("fetch_arxiv")
def fetch_arxiv(query: str):
    """Fetch relevant papers from ArXiv based on a query."""
    papers = fetch_arxiv_papers(query)
    if isinstance(papers, str):  # Error message
        return papers
    # Return only title and link, without the abstract
    return "\n---\n".join(
        [f"Title: {title}\nLink: {link}" for title, _, link in papers]
    )

@tool("final_answer")
def final_answer(
    introduction: str,
    research_steps: str,
    main_body: str,
    conclusion: str,
    sources: str
):
    """返回一个包含研究报告的自然语言响应。"""
    return ""

# 定义系统提示和提示模板
system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (ie, if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad) use the final_answer
tool."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}"),
])

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

tools = [
    rag_search_filter,
    web_search,
    fetch_arxiv,
    final_answer
]

def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

manager = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(
            intermediate_steps=x["intermediate_steps"]
        ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[AgentAction], operator.add]

# 定义运行管理器的函数
def run_manager(state: dict):
    out = manager.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    state['intermediate_steps'].append(action_out)
    return state

def router(state: dict):
    if isinstance(state["intermediate_steps"], list):
        # Use fetch_arxiv if the query suggests an ArXiv-related question
        if "arxiv" in state["input"].lower() or "research papers" in state["input"].lower():
            return "fetch_arxiv"
        return state["intermediate_steps"][-1].tool
    else:
        return "final_answer"

tool_str_to_func = {
    "rag_search_filter": rag_search_filter,
    "web_search": web_search,
    "fetch_arxiv": fetch_arxiv,
    "final_answer": final_answer
}

def run_tool(state: dict):
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    tool_func = tool_str_to_func.get(tool_name)

    if not tool_func:
        raise ValueError(f"Tool {tool_name} not found")

    try:
        out = tool_func.invoke(input=tool_args)
        if not out or out == "No relevant documents found in Pinecone. Consider expanding your query.":
            if tool_name == "rag_search_filter":
                out = web_search(tool_args["query"])  # Fallback to web search
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in tool {tool_name}: {str(e)}")

    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    state['intermediate_steps'].append(action_out)
    return state



graph = StateGraph(AgentState)
graph.add_node("manager", run_manager)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("fetch_arxiv", run_tool)
graph.add_node("final_answer", run_tool)
graph.set_entry_point("manager")

graph.add_conditional_edges(
    source="manager",
    path=router,
)

for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "manager")

graph.add_edge("final_answer", END)

runnable = graph.compile()

# 定义请求模型
class QueryRequest(BaseModel):
    input: str

# 定义响应模型
class QueryResponse(BaseModel):
    introduction: str
    research_steps: str
    main_body: str
    conclusion: str
    sources: str
    arxiv_results: list = []

def build_report(output: dict):
    research_steps = output.get("research_steps", "")
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    sources = output.get("sources", "")
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])
    return {
        "introduction": output.get("introduction", ""),
        "research_steps": research_steps,
        "main_body": output.get("main_body", ""),
        "conclusion": output.get("conclusion", ""),
        "sources": sources
    }

@app.get("/publications")
async def get_publications():
    # Since Pinecone doesn't support listing unique metadata values directly,
    # we'll retrieve a sample of vectors to extract publications.
    query_result = index.query(vector=[0]*dims, top_k=1000, include_metadata=True)
    publications = set()
    for match in query_result['matches']:
        metadata = match.get('metadata', {})
        publication = metadata.get('publication')
        if publication:
            publications.add(publication)
    return {"publications": sorted(publications)}

# 定义 FastAPI 路由
@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    initial_state = {
        "input": request.input,
        "chat_history": [],
        "intermediate_steps": []
    }
    arxiv_results = []
    
    try:
        # Run the main query and generate a report
        final_state = runnable.invoke(initial_state)
        last_action = final_state["intermediate_steps"][-1]

        if last_action.tool == "final_answer":
            report = build_report(last_action.tool_input)
            
            # Fetch ArXiv results if enabled
            arxiv_papers = fetch_arxiv(request.input)  # Directly call fetch_arxiv tool
            if arxiv_papers:
                arxiv_results = arxiv_papers.split("\n---\n")  # Split ArXiv results for easier processing
            
            # Include ArXiv results in the final response
            return {**report, "arxiv_results": arxiv_results}
        else:
            raise HTTPException(status_code=500, detail="Failed to generate final answer")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))