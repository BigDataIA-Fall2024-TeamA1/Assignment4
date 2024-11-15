# data_processing.py

import os
import shutil  # Standard library imports can remain at the module level
import time

def download_publications(**kwargs):
    from dotenv import load_dotenv
    load_dotenv()
    import boto3

    # Access AWS credentials from environment variables
    AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.environ.get('AWS_REGION')
    AWS_BUCKET = os.environ.get('AWS_BUCKET')

    s3 = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # Specify the folder in S3 bucket
    s3_folder = 'pdfs/'
    local_download_path = '/tmp/downloaded_pdfs'

    if not os.path.exists(local_download_path):
        os.makedirs(local_download_path)

    # List objects in the specified S3 folder
    response = s3.list_objects_v2(Bucket=AWS_BUCKET, Prefix=s3_folder, MaxKeys=2)

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            if key.endswith('.pdf'):
                file_name = os.path.basename(key)
                local_file_path = os.path.join(local_download_path, file_name)
                s3.download_file(AWS_BUCKET, key, local_file_path)
                print(f'Downloaded {file_name}')
    else:
        print('No PDFs found in the specified S3 folder.')

def parsing_publications(**kwargs):
    import traceback
    from docling.document_converter import DocumentConverter

    # Paths
    pdf_folder = '/tmp/downloaded_pdfs'
    markdown_output_folder = '/tmp/markdown_files'

    if not os.path.exists(markdown_output_folder):
        os.makedirs(markdown_output_folder)

    converter = DocumentConverter()

    pdf_files = os.listdir(pdf_folder)
    for pdf_file in pdf_files:
        if pdf_file.endswith('.pdf'):
            if pdf_file == 'rf-v2013-n4-1-pdf.pdf':
                print(f"Skipping {pdf_file} due to known issue.")
                continue

            print(f"Processing {pdf_file}...")
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                result = converter.convert(pdf_path)
                markdown_content = result.document.export_to_markdown()

                markdown_file_name = pdf_file.replace('.pdf', '.md')
                markdown_file_path = os.path.join(markdown_output_folder, markdown_file_name)

                with open(markdown_file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f'Parsed {pdf_file} to {markdown_file_name}')
            except Exception as e:
                print(f'Error parsing {pdf_file}: {e}')
                traceback.print_exc()

def embedding_and_upload_pinecone(**kwargs):
    from openai import OpenAI
    from pinecone import Pinecone, ServerlessSpec

    # Access OpenAI and Pinecone credentials from environment variables
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')
    PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT', 'us-east1-gcp')

    # Validate API keys
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY environment variable not set.")

    # Initialize OpenAI client
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return

    index_name = PINECONE_INDEX_NAME

    # Delete existing index if it exists
    try:
        if index_name in pc.list_indexes():
            print(f"Deleting existing index: {index_name}")
            pc.delete_index(index_name)
            time.sleep(1)  # Wait for the index to be deleted
    except Exception as e:
        print(f"Error deleting existing index: {e}")
        return

    # Create a new index with the correct dimension
    try:
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',  # or 'gcp', depending on your Pinecone setup
                region='us-east-1'  # replace with your desired region
            )
        )
    except Exception as e:
        print(f"Error creating new index: {e}")
        return

    try:
        index = pc.Index(index_name)
    except Exception as e:
        print(f"Error accessing index: {e}")
        return

    # Paths
    markdown_folder = '/tmp/markdown_files'

    # Function to get embeddings using OpenAI's API
    def get_embedding(text):
        try:
            response = client.embeddings.create(
                input=text,
                model='text-embedding-ada-002'  # Use the latest embedding model
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536  # Return a zero vector in case of error

    # Set batch size
    BATCH_SIZE = 100  # You can adjust this value based on your needs

    for md_file in os.listdir(markdown_folder):
        if md_file.endswith('.md'):
            md_file_path = os.path.join(markdown_folder, md_file)
            with open(md_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Split content into paragraphs
            texts = markdown_content.split('\n\n')

            embeddings = []
            for text in texts:
                embedding = get_embedding(text)
                embeddings.append(embedding)
                time.sleep(0.2)  # Add delay to respect rate limits

            # Prepare data for Pinecone
            vectors = []
            publication_name = md_file.replace('.md', '')
            for idx, embedding in enumerate(embeddings):
                vector_id = f"{publication_name}_{idx}"
                metadata = {
                    'publication': publication_name,
                    'source': md_file,
                    'text': texts[idx]
                }
                vectors.append((vector_id, embedding, metadata))

            # Upsert vectors to Pinecone in batches
            for i in range(0, len(vectors), BATCH_SIZE):
                batch = vectors[i:i+BATCH_SIZE]
                try:
                    index.upsert(vectors=batch)
                    print(f'Uploaded batch {i//BATCH_SIZE + 1} of embeddings for {md_file}')
                except Exception as e:
                    print(f"Error uploading batch {i//BATCH_SIZE + 1} for {md_file}: {str(e)}")
                    # Implement retry mechanism if needed

            print(f'Finished uploading all embeddings for {md_file}')

def delete_local_publications(**kwargs):
    import shutil  # If not imported at the module level
    # Define the paths used in the previous tasks
    pdf_folder = '/tmp/downloaded_pdfs'
    markdown_folder = '/tmp/markdown_files'

    # Delete the PDF folder
    if os.path.exists(pdf_folder):
        shutil.rmtree(pdf_folder)
        print(f'Deleted folder {pdf_folder}')

    # Delete the markdown files folder
    if os.path.exists(markdown_folder):
        shutil.rmtree(markdown_folder)
        print(f'Deleted folder {markdown_folder}')

def main():
    print("Starting download of publications...")
    download_publications()
    
    print("Parsing downloaded publications...")
    parsing_publications()
    
    print("Embedding and uploading to Pinecone...")
    embedding_and_upload_pinecone()
    
    print("Cleaning up local files...")
    delete_local_publications()

if __name__ == "__main__":
    main()