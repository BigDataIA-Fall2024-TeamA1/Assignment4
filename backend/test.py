import os
import time
from getpass import getpass
from dotenv import load_dotenv
from langchain_core.tools import tool
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize OpenAI encoder
encoder = OpenAIEncoder(name="text-embedding-3-small")

# Initialize Pinecone
api_key = os.getenv("PINECONE_API_KEY") or getpass("Pinecone API key: ")
pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud="aws", region="us-east-1")
dims = len(encoder(["some random text"])[0])
index_name = "publications-vectors"

# Ensure the index exists
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

def web_search(query: str):
    """Fallback web search stub."""
    return f"Fallback: Web search result for query '{query}'"

@tool("rag_search_filter")
def rag_search_filter(query: str, publication: str):
    """Test RAG search filter tool."""
    xq = encoder([query])  # Generate vector embedding for the query
    print(f"Query vector dimensions: {len(xq[0])}")
    
    # Perform similarity search in Pinecone
    xc = index.query(vector=xq, top_k=6, include_metadata=True, filter={"publication": publication})
    print(f"Pinecone Query Results: {xc}")
    
    # Check for matches
    if not xc["matches"]:
        print("No matches found in Pinecone. Fallback to web search.")
        return web_search(query)
    
    # Format the results
    context_str = format_rag_contexts(xc["matches"])
    return context_str

# Test the function
if __name__ == "__main__":
    query = input("Enter your query: ")
    publication = input("Enter publication filter (or leave empty for no filter): ")

    if not publication:
        publication = None

    # Use `invoke` instead of deprecated `__call__`
    result = rag_search_filter.invoke({"query": query, "publication": publication})
    print("\nResult:\n", result)
