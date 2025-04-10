# Optimized Document Analysis and Q&A System

This document outlines the improvements made to the original Document Analysis and Q&A System to reduce Tokens Per Minute (TPM) usage while maintaining the accuracy of responses. The optimizations focus on efficient token usage, streamlined agent interactions, and enhanced context sharing.

## Key Improvements

### 1. Optimized Token Usage
- **Chunk strategies and embedd chunks**
    - chunk the document in chunks of 500 char with 50 overlapping chars
    - embedd using `text-embedding-ada-002`
- **Shared Context for Related Questions:**
  - Introduced a mechanism to group related questions and reuse document chunks, reducing redundant embedding calls.
  - Cached relevant document chunks for subsequent questions in the same group.

### 2. Streamlined Agent Interactions
- **Simplified Agent Setup:**
  - Created a single combined agent (`AgentGroupChat`) for complex questions, reducing the overhead of managing multiple agents.
  - Limited the number of iterations for agent responses to save tokens.
- **Improved Validation Process:**
  - Enhanced the `Validator` agent to provide concise feedback and validate answers in fewer steps.

### 3. Enhanced Context Sharing
- **Grouped Questions:**
  - Grouped related questions based on field names to share context and avoid redundant processing.
  - Used shared context to answer multiple questions efficiently.
- **Efficient Document Chunking:**
  - Cached document chunks for reuse across related questions, reducing the need for repeated document retrieval and embedding generation.


## Updated Architecture

The updated architecture focuses on efficient token usage, streamlined agent interactions, and embedding-based document processing. Below is a diagram illustrating the optimized approach:

![Optimized Architecture Diagram](optimized_architecture.jpg)

### Key Changes in the Architecture

1. **Document Embedding Process:**
   - Documents are chunked into smaller sections (500 characters with 50 overlapping characters).
   - Each chunk is embedded using the `text-embedding-ada-002` model.
   - Embeddings are cached and reused for related questions to reduce redundant processing.

2. **Combined Agent for Complex Questions:**
   - A single `AgentGroupChat` is used for complex questions.
   - Includes two sub-agents:
     - **Financial Expert Agent:** Extracts and formats answers based on the provided context.
     - **Validator Agent:** Validates the answers for accuracy and formatting.
   - Reduces the overhead of managing multiple specialized agents.

3. **Document Retriever Agent:**
   - Retrieves relevant document chunks based on the embeddings.
   - Uses cached embeddings to improve efficiency and reduce token usage.

4. **Shared Context for Related Questions:**
   - Groups related questions to share context and avoid redundant processing.
   - Uses cached document chunks for subsequent questions in the same group.

## Benefits of the Optimized System
- **Reduced Costs:**
  - Lower token usage leads to reduced operational costs.
- **Improved Efficiency:**
  - Faster response times due to shared context and streamlined agent interactions.
- **Maintained Accuracy:**
  - Ensures high-quality answers with accurate validation and formatting.

## Limitations
- The system is still optimized primarily for financial documents.
- Complex layouts may require additional processing steps.

## How to Use the Optimized System

### Prerequisites
- Azure Document Intelligence service
- Azure OpenAI service

### Environment Configuration
1. Create a `.env` file with your Azure service credentials and embedding model configuration:

```
DOCUMENTINTELLIGENCE_ENDPOINT=your_document_intelligence_endpoint
DOCUMENTINTELLIGENCE_API_KEY=your_document_intelligence_key
AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_API_VERSION=2023-05-15
EMBEDDING_MODEL=text-embedding-ada-002
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Processing Documents
To process a collection of documents and generate embeddings:

```python
from document_processing import process_folder
from generate_embeddings import create_document_embeddings

# Step 1: Process all PDFs in the specified folder and save results to processed_documents/
process_folder("anonymized_examples")

# Step 2: Generate embeddings for the processed documents
create_document_embeddings("processed_documents/document_index.json")
```

### Schema-Based Q&A
The system uses JSON schema files to define questions and answer formats. Here's an example schema:

```json
{
  "assetsAndIncome": {
    "properties": {
      "totalBankableAssets": {
        "displayName": "Total Bankable Assets",
        "description": "What is the total value of bankable assets for $_var_individual?",
        "example": "CHF 1'500'000.00"
      },
      "totalRealEstateValue": {
        "displayName": "Real Estate Value",
        "description": "What is the total value of real estate owned by $_var_individual?",
        "example": "CHF 2'500'000.00"
      }
    }
  }
}
```

## Summary
The optimized system significantly reduces token usage while maintaining the accuracy and quality of responses. By leveraging shared context, streamlined agent interactions, and efficient document processing, the system achieves better performance and cost efficiency.
