import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import time

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from openai import AzureOpenAI
from dotenv import load_dotenv

from document_embedding import retrieve_relevant_chunks, create_document_embeddings

load_dotenv()

# Azure OpenAI configuration
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]


# Initialize direct client for simple questions
client = AzureOpenAI(
    api_key=aoai_key,
    api_version=api_version,
    azure_endpoint=aoai_endpoint
)

def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()
    kernel.add_service(service=AzureChatCompletion())
    return kernel

def load_document_index(index_path: str = "processed_documents/document_index.json") -> List[Dict[str, Any]]:
    """Load the document index containing metadata for all processed documents."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_form_schema(schema_path: str = "schema/schema_assetsAndIncome.json") -> Dict[str, Any]:
    """Load the form schema containing questions to answer."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def group_related_questions(properties: Dict) -> Dict[str, List]:
    """Group related questions based on their field names to share context."""
    groups = {}
    for field_name, question in properties.items():
        # Extract base field name (e.g., 'bankableAssets' from 'bankableAssetsComments')
        base_name = field_name.replace("Comments", "")
        
        if base_name not in groups:
            groups[base_name] = []
        
        groups[base_name].append((field_name, question))
    
    return groups
    
async def process_question_optimized(
    chat: Optional[AgentGroupChat], 
    question: Dict[str, Any], 
    document_index: List[Dict[str, Any]],
    var_individual: str,
    cached_chunks: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    """Process a single question with optimized token usage."""
    question_text = question["description"].replace("$_var_individual", var_individual)
    display_name = question["displayName"]
    example = question["example"]
    
    print(f"\n\n--- Processing: {display_name} ---")
    print(f"Question: {question_text}")
    
    # Use cached chunks if provided, otherwise retrieve relevant chunks
    top_k = 3
    complexity = "complex"
    if cached_chunks is None:
        relevant_chunks = retrieve_relevant_chunks(question_text, document_index, top_k=top_k)
    else:
        relevant_chunks = cached_chunks
        print("Using cached document chunks")
    
    if not relevant_chunks:
        return {
            "field_name": display_name,
            "question": question_text,
            "answer": "No relevant information found in documents.",
            "sources": []
        }
    
    # Compile relevant content from chunks
    relevant_content = "\n\n".join([
        f"--- From document: {chunk['document']} ---\n{chunk['content']}"
        for chunk in relevant_chunks
    ])
    
    document_paths = [chunk["document_path"] for chunk in relevant_chunks]
    
    # For complex questions, use the agent-based approach
    if chat is None:
        print("No chat agent provided for complex question")
        return {
            "field_name": display_name,
            "question": question_text,
            "answer": "Error: Chat agent required for complex questions",
            "sources": []
        }
    
    # Reset chat for a new question
    await chat.reset()
    
    # Add the question to the chat
    await chat.add_chat_message(
        f"Question: {question_text}\n\n"
        f"FORMAT REQUIREMENT: Your answer must follow exactly this example format: \"{example}\"\n"
        f"Pay special attention to currency notation, spacing, and structure.\n\n"
        f"Relevant document content:\n{relevant_content}"
    )
    
    # Process conversation for this question
    answer = None
    
    async for response in chat.invoke():
        if response is None or not response.name:
            continue
        
        print(f"\n# {response.name}:")
        print(response.content)
        
        # When answer validator confirms the answer, extract the final answer
        if "ANSWER VALIDATED" in response.content:
            content = response.content
            if "FINAL ANSWER:" in content:
                answer_start = content.find("FINAL ANSWER:") + len("FINAL ANSWER:")
                answer = content[answer_start:].strip()
                break
    
    # Check if we got a valid answer
    if not answer:
        answer = "Unable to determine answer from available documents."
    
    return {
        "field_name": display_name,
        "question": question_text,
        "answer": answer,
        "sources": document_paths,
        "context": relevant_content
    }

async def create_optimized_agent(kernel: Kernel, var_individual: str) -> AgentGroupChat:
    """Create a streamlined agent for complex questions."""
    # Create a single combined agent for complex questions
    from semantic_kernel.agents.strategies import DefaultTerminationStrategy
    
    # Create agent with focused, simplified instructions
    financial_agent = ChatCompletionAgent(
        kernel=kernel,
        name="FinancialExpert",
        instructions=f"""
You are a financial data extraction expert that answers specific questions about {var_individual}'s financial information.

Process:
1. Analyze the query to understand what financial information is needed
2. Review the provided document content (which will be limited to only relevant sections)
3. Extract the exact information requested
4. Format your response according to the example format provided with each question
5. Be concise and precise with your answers

Format currency values consistently (e.g., "CHF 1'000'000.00")
If the information is not in the provided context, respond with "Information not available in provided documents".
"""
    )
    
    # Create a validator with simplified instructions
    validator_agent = ChatCompletionAgent(
        kernel=kernel,
        name="Validator",
        instructions=f"""
You are a validator for financial information extraction. Check if answers are correctly formatted according to examples.

If the answer needs improvement: Provide SPECIFIC feedback on what needs to be fixed

If the answer is complete and accurate: Format your response EXACTLY as follows:
ANSWER VALIDATED: [Brief reason why the answer is good]

FINAL ANSWER:
[The complete final answer, formatted EXACTLY according to the example response]
"""
    )
    
    # Create a history reducer to keep context manageable
    history_reducer = ChatHistoryTruncationReducer(target_count=4)
    
    # Create a simplified chat setup with just two agents
    return AgentGroupChat(
        agents=[financial_agent, validator_agent],
        termination_strategy=DefaultTerminationStrategy(
            maximum_iterations=3  # Reduce maximum iterations to save tokens
        ),
        chat_history=history_reducer,
    )

async def process_form_schema_optimized(
    schema_path: str, 
    var_individual: str = "Hans Muster"
) -> Dict[str, Any]:
    """Process all questions in the form schema with optimized token usage."""
    
    # Load necessary data
    schema = load_form_schema(schema_path)
    document_index = load_document_index()
    
    # Get the category name and questions
    category_name = list(schema.keys())[0]  # e.g., "assetsAndIncome"
    properties = schema[category_name]["properties"]
    
    # Create kernel once for all questions
    kernel = create_kernel()
    
    # Group related questions to share context
    question_groups = group_related_questions(properties)
    
    # Process each group of questions
    results = {}
    
    for base_field, questions in question_groups.items():
        print(f"\n=== Processing question group: {base_field} ===")
        
        # Get first question in the group
        first_field, first_question = questions[0]
        
        # Create a fresh chat agent for this question group
        complex_chat = await create_optimized_agent(kernel, var_individual)
        
        # Process first question in group to get shared context
        result = await process_question_optimized(
            complex_chat,
            first_question, 
            document_index, 
            var_individual
        )
        
        results[first_field] = result["answer"]
        
        # Use the same chunks for related questions to avoid redundant embedding calls
        cached_chunks = None
        if "context" in result and len(questions) > 1:
            # Create fake chunks from the context to reuse
            relevant_content = result["context"]
            cached_chunks = [{
                "document": doc_path.split("/")[-1],
                "document_path": doc_path,
                "content": relevant_content,
                "similarity": 1.0
            } for doc_path in result["sources"]]
        
        # Process additional questions in the group
        for field_name, question in questions[1:]:
            # Skip the first question as we already processed it
            if field_name == first_field:
                continue
            
            # Create a fresh chat agent for each question to avoid the "agent is active" error
            complex_chat = await create_optimized_agent(kernel, var_individual)
                
            # Process with shared context
            related_result = await process_question_optimized(
                complex_chat,
                question, 
                document_index, 
                var_individual,
                cached_chunks
            )
            
            results[field_name] = related_result["answer"]
    
    # Format the output according to schema
    output = {
        category_name: {}
    }
    
    for field_name, answer in results.items():
        output[category_name][field_name] = answer
    
    return output

async def main():
    # Set the individual name to look for in documents
    var_individual = "Hans Muster"
    
    # Path to the schema file
    schema_path = "schema/schema_assetsAndIncome.json"
    
    print(f"Processing form for individual: {var_individual}")
    print(f"Using schema: {schema_path}")
    print("-" * 50)
    
    # Track token usage
    start_time = time.time()
    
    # Process the schema and generate answers
    results = await process_form_schema_optimized(schema_path, var_individual)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Save results to file
    output_path = f"{var_individual.replace(' ', '_')}_form_answers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    
    # Print a summary of results
    print("\nSummary of answers:")
    for field, answer in results["assetsAndIncome"].items():
        print(f"{field}: {answer[:50]}..." if len(answer) > 50 else answer)

if __name__ == "__main__":
    asyncio.run(main())
