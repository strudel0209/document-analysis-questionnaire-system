import os
import json
import numpy as np
from typing import List, Dict, Any
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI configuration
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_key = os.environ["AZURE_OPENAI_API_KEY"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]
embedding_deployment = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"]

def chunk_document(content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split document content into overlapping chunks."""
    if not content:
        return []
        
    # Split content into paragraphs
    paragraphs = content.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds chunk size and we already have content
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Keep some overlap for context
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
        
        current_chunk += para + "\n\n"
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def create_embeddings(text: str, client) -> List[float]:
    """Create embeddings for a text using Azure OpenAI."""
    response = client.embeddings.create(
        input=text,
        model=embedding_deployment  # Use the deployment name from environment variables
    )
    return response.data[0].embedding

def create_document_embeddings(doc_path: str) -> Dict[str, Any]:
    """Process a document by chunking it and generating embeddings."""
    client = AzureOpenAI(
        api_key=aoai_key,
        api_version=api_version,
        azure_endpoint=aoai_endpoint
    )
    
    # Extract base filename without extension
    content_path = doc_path.replace("_processed.json", "_content.md")
    embedding_path = doc_path.replace("_processed.json", "_embeddings.json")
    
    # Check if embeddings already exist
    if os.path.exists(embedding_path):
        with open(embedding_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Load document metadata
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
    
    # Load document content
    with open(content_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Chunk the document
    chunks = chunk_document(content)
    
    # Create embeddings for each chunk
    chunk_data = []
    for i, chunk in enumerate(chunks):
        embedding = create_embeddings(chunk, client)
        chunk_data.append({
            "chunk_id": i,
            "content": chunk,
            "embedding": embedding
        })
    
    # Save chunks and embeddings
    embedding_data = {
        "file_path": doc_data["file_path"],
        "file_name": doc_data["file_name"],
        "metadata": doc_data["metadata"],
        "chunks": chunk_data
    }
    
    with open(embedding_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_data, f)
    
    return embedding_data

def vector_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def retrieve_relevant_chunks(query: str, document_index: List[Dict], top_k: int = 3) -> List[Dict]:
    """Retrieve most relevant chunks using semantic similarity."""
    client = AzureOpenAI(
        api_key=aoai_key,
        api_version=api_version,
        azure_endpoint=aoai_endpoint
    )
    
    # Create query embedding
    query_embedding = create_embeddings(query, client)
    
    all_chunks = []
    
    # First, ensure we have embeddings for all documents
    for doc in document_index:
        doc_path = doc["processed_path"]
        embedding_path = doc_path.replace("_processed.json", "_embeddings.json")
        
        # Generate embeddings if they don't exist
        if not os.path.exists(embedding_path):
            create_document_embeddings(doc_path)
    
    # Now retrieve the most relevant chunks
    for doc in document_index:
        doc_path = doc["processed_path"]
        embedding_path = doc_path.replace("_processed.json", "_embeddings.json")
        
        with open(embedding_path, 'r', encoding='utf-8') as f:
            embedding_data = json.load(f)
        
        # Calculate similarity for each chunk
        for chunk in embedding_data["chunks"]:
            similarity = vector_similarity(query_embedding, chunk["embedding"])
            all_chunks.append({
                "document": doc["metadata"]["title"],
                "document_path": doc_path,
                "content": chunk["content"],
                "similarity": similarity
            })
    
    # Sort chunks by similarity and return top_k
    sorted_chunks = sorted(all_chunks, key=lambda x: x["similarity"], reverse=True)
    return sorted_chunks[:top_k]   # default to top 3
