import os
import boto3
from docling.document_converter import DocumentConverter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import shutil
from dotenv import load_dotenv

load_dotenv()

def download_publications(**kwargs):
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
    # Access Pinecone credentials from environment variables
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    if PINECONE_INDEX_NAME not in pc.list_indexes():
        print(f"Creating new index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,  # Adjust dimension as per your model
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'  # Choose an appropriate region
            )
        )
    else:
        print(f"Index {PINECONE_INDEX_NAME} already exists. Using existing index.")

    index = pc.Index(PINECONE_INDEX_NAME)

    # Paths
    markdown_folder = '/tmp/markdown_files'

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose a different model if needed

    # Set batch size
    BATCH_SIZE = 100  # You can adjust this value based on your needs

    for md_file in os.listdir(markdown_folder):
        if md_file.endswith('.md'):
            md_file_path = os.path.join(markdown_folder, md_file)
            with open(md_file_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()

            # Split content into paragraphs
            texts = markdown_content.split('\n\n')

            embeddings = model.encode(texts)

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
                vectors.append((vector_id, embedding.tolist(), metadata))

            # Upsert vectors to Pinecone in batches
            for i in range(0, len(vectors), BATCH_SIZE):
                batch = vectors[i:i+BATCH_SIZE]
                try:
                    index.upsert(vectors=batch)
                    print(f'Uploaded batch {i//BATCH_SIZE + 1} of embeddings for {md_file}')
                except Exception as e:
                    print(f"Error uploading batch {i//BATCH_SIZE + 1} for {md_file}: {str(e)}")
                    # You might want to implement a retry mechanism here

            print(f'Finished uploading all embeddings for {md_file}')

    # No need to close the index

def delete_local_publications(**kwargs):
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

# def main():
#     print("Starting download of publications...")
#     download_publications()
    
#     print("Parsing downloaded publications...")
#     parsing_publications()
    
#     print("Embedding and uploading to Pinecone...")
#     embedding_and_upload_pinecone()
    
#     print("Cleaning up local files...")
#     delete_local_publications()

# if __name__ == "__main__":
#     main()
