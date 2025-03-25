import os
import json
from azure.core.exceptions import HttpResponseError
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat, AnalyzeResult
from openai import AzureOpenAI

load_dotenv()

# Document Intelligence configuration
endpoint = os.environ["DOCUMENTINTELLIGENCE_ENDPOINT"]
key = os.environ["DOCUMENTINTELLIGENCE_API_KEY"]

# Azure OpenAI configuration
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]

def analyze_local_document_to_markdown(file_path):
    """Process a local PDF file and convert to markdown format."""
    
    document_intelligence_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    # Open and read the local file
    with open(file_path, "rb") as f:
        file_content = f.read()
    
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout",
        file_content,  # Pass the binary file content directly
        output_content_format=DocumentContentFormat.MARKDOWN,
        content_type="application/pdf"  # Specify the content type
    )
    
    result: AnalyzeResult = poller.result()
    
    print(f"Processed {file_path}")
    print(f"Content format: {result.content_format}")
    return result.content

def generate_metadata_with_llm(markdown_content, filename):
    """Generate document metadata using Azure OpenAI."""
    client = AzureOpenAI(
        api_key=aoai_key,
        api_version=api_version,
        azure_endpoint=aoai_endpoint
    )
    
    # Limit content length to avoid token limits
    content_sample = markdown_content[:6000] + ("..." if len(markdown_content) > 6000 else "")
    
    system_prompt = """You are a document metadata specialist. 
    Analyze the document content and generate comprehensive metadata in JSON format.
    Include only the JSON in your response, with no additional text."""
    
    user_prompt = f"""
    Based on the following document content from file '{filename}', generate metadata in JSON format:

    DOCUMENT CONTENT:
    {content_sample}

    Generate JSON with the following fields:
    1. "title": A concise document title
    2. "description": A brief description (2-3 sentences)
    3. "summary": A comprehensive summary (5-6 sentences covering key topics)
    4. "document_type": The type of document (report, invoice, letter, etc.)
    5. "key_entities": List of important named entities (people, companies, places)
    6. "categories": List of relevant categories/topics
    """
    
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=800
    )
    
    metadata_text = response.choices[0].message.content
    
    # Extract JSON from response (in case there's surrounding text)
    try:
        # Find JSON content - look for content between curly braces
        import re
        json_match = re.search(r'\{.*\}', metadata_text, re.DOTALL)
        if json_match:
            metadata_text = json_match.group(0)
        
        metadata = json.loads(metadata_text)
        return metadata
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON metadata: {e}")
        print(f"Raw response: {metadata_text}")
        # Fallback metadata
        return {
            "title": os.path.basename(filename),
            "description": "Document extracted from PDF",
            "summary": "Content summary unavailable",
            "document_type": "unknown",
            "key_entities": [],
            "categories": []
        }

def process_document_with_metadata(file_path):
    """Process a document and generate both markdown content and metadata."""
    # Get markdown content
    markdown_content = analyze_local_document_to_markdown(file_path)
    
    # Generate metadata
    filename = os.path.basename(file_path)
    metadata = generate_metadata_with_llm(markdown_content, filename)
    
    # Combine into a single result object
    result = {
        "file_path": file_path,
        "file_name": filename,
        "metadata": metadata,
        "markdown_content": markdown_content
    }
    
    return result

def find_all_pdfs(folder_path):
    """Find all PDF files in folder_path, including in nested folders."""
    pdf_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def load_existing_document_index(output_dir):
    """Load existing document index if it exists."""
    index_path = os.path.join(output_dir, "document_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing document index: {e}")
    return []

def get_already_processed_paths(processed_files):
    """Extract the file paths that have already been processed."""
    return {doc["original_path"] for doc in processed_files}

def process_folder(folder_path, output_dir="processed_documents", force_reprocess=False):
    """Process all PDF files in a folder and its subfolders."""
    # Find all PDF files
    pdf_files = find_all_pdfs(folder_path)
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load existing document index
    existing_processed_files = load_existing_document_index(output_dir)
    already_processed = get_already_processed_paths(existing_processed_files)
    
    # Check how many files need processing
    new_files = [f for f in pdf_files if f not in already_processed]
    if not force_reprocess:
        files_to_process = new_files
        print(f"Found {len(existing_processed_files)} previously processed files")
        print(f"Will process {len(files_to_process)} new files")
    else:
        files_to_process = pdf_files
        print(f"Force reprocessing enabled. Will process all {len(files_to_process)} files")
    
    # Track processed files
    processed_files = existing_processed_files.copy() if not force_reprocess else []
    
    # Process each PDF file
    for i, file_path in enumerate(files_to_process):
        try:
            print(f"\n[{i+1}/{len(pdf_files)}] Processing: {file_path}")
            result = process_document_with_metadata(file_path)
            
            # Create relative path structure in output directory
            rel_path = os.path.relpath(os.path.dirname(file_path), folder_path)
            file_output_dir = os.path.join(output_dir, rel_path)
            os.makedirs(file_output_dir, exist_ok=True)
            
            # Save results
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(file_output_dir, f"{base_name}_processed.json")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                # Save just the metadata and file reference, not the full content
                output_data = {
                    "file_path": file_path,
                    "file_name": result['file_name'],
                    "metadata": result['metadata'],
                    # Save markdown to separate file to keep JSON file smaller
                    "markdown_file": f"{base_name}_content.md"
                }
                json.dump(output_data, f, indent=2)
            
            # Save markdown content separately
            markdown_path = os.path.join(file_output_dir, f"{base_name}_content.md")
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(result['markdown_content'])
            
            print(f"Saved to {output_path}")
            processed_files.append({
                "original_path": file_path,
                "processed_path": output_path,
                "metadata": result['metadata']
            })
            
        except HttpResponseError as error:
            print(f"Error processing {file_path}: {error.message}")
        except Exception as e:
            print(f"Unexpected error processing {file_path}: {e}")
    
    # Create an index file with all processed documents
    index_path = os.path.join(output_dir, "document_index.json")
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(processed_files, f, indent=2)
    
    print(f"\nProcessing complete. Processed {len(processed_files)} of {len(pdf_files)} files.")
    print(f"Document index saved to {index_path}")
    
    # return processed_files
    for doc in processed_files:
        print(f"- {doc['original_path']} → {doc['metadata']['title']}")