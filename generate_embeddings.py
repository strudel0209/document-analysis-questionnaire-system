from document_embedding import create_document_embeddings
import json
import os

# Ensure processed_documents directory exists
if not os.path.exists('processed_documents/document_index.json'):
    print("Error: document_index.json not found. Please run document processing first.")
    exit(1)

# Load document index
with open('processed_documents/document_index.json', 'r') as f:
    docs = json.load(f)

# Process each document
print(f"Found {len(docs)} documents to process")
for i, doc in enumerate(docs):
    print(f"[{i+1}/{len(docs)}] Processing: {os.path.basename(doc['processed_path'])}")
    create_document_embeddings(doc['processed_path'])

print("Embedding generation complete!")
