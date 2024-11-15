import os
from pinecone import Pinecone

# Load environment variables
api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Check if environment variables are set
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
if not index_name:
    raise ValueError("PINECONE_INDEX_NAME not found in environment variables.")

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=api_key)
    print("Successfully connected to Pinecone with the provided API key.")
except Exception as e:
    print("Failed to connect to Pinecone. Please check your API key.")
    print("Error:", e)
    exit(1)

# List all available indexes and extract names
try:
    available_indexes_response = pc.list_indexes()
    available_indexes = [index['name'] for index in available_indexes_response.get('indexes', [])]
    print("Available indexes in Pinecone:", available_indexes)

    # Check if the specified index exists
    if index_name in available_indexes:
        print(f"Index '{index_name}' exists in Pinecone.")
        
        # Optionally, get and print details about the index
        index = pc.Index(index_name)
        index_stats = index.describe_index_stats()
        print("Index stats:", index_stats)
    else:
        print(f"Index '{index_name}' does not exist in Pinecone. Please check the index name.")
except Exception as e:
    print("An error occurred while accessing the indexes.")
    print("Error:", e)
