import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import time
import re

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
    DefaultTerminationStrategy
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt, kernel_function
from openai import AzureOpenAI
from dotenv import load_dotenv

from document_embedding import retrieve_relevant_chunks, create_document_embeddings

load_dotenv()

# Azure OpenAI configuration
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_key = os.environ["AZURE_OPENAI_API_KEY"]
deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"]
api_version = os.environ["AZURE_OPENAI_API_VERSION"]

# Define agent names
DOCUMENT_RETRIEVER = "DocumentRetriever"
FINANCIAL_EXPERT = "FinancialExpert"
ANSWER_VALIDATOR = "AnswerValidator"

# Token usage optimization settings
MAX_CHUNK_SIZE = 1500  # Maximum content size to send to agents
MAX_CHUNKS_PER_CALL = 3  # Maximum number of document chunks to process
MAX_ITERATIONS = 3  # Maximum conversation iterations
TRUNCATE_HISTORY = 4  # Keep only top messages in history

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

def truncate_text(text: str, max_length: int = MAX_CHUNK_SIZE) -> str:
    """Truncate text to specified maximum length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def extract_currency_amount(text: str) -> Optional[str]:
    """Extract currency amounts from text in CHF format."""
    pattern = r"CHF\s*(\d{1,3}(?:'\d{3})*(?:\.\d{2})?)"
    match = re.search(pattern, text)
    return f"CHF {match.group(1)}" if match else None

class DocumentContentPlugin:
    def _load_document_content(self, path: str) -> Optional[str]:
        """Loads document content from a file if it exists."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                return file.read()
        return None
    
    @kernel_function
    async def retrieve_relevant_chunks(
        self,
        query: str,
        top_k: int = 4,  # Reduced from 8 to limit token usage
        individual_name: str = "Hans Muster"
    ) -> str:
        """
        Retrieve the most semantically relevant document chunks for a question.
        Uses embedding-based retrieval to find content across all documents.
        """
        try:
            # Load document index
            document_index = load_document_index()
            all_relevant_chunks = []

            # Process only a subset of documents to limit token usage
            processed_docs = 0
            max_docs = 10  # Limit the number of documents to process
            
            for document in document_index:
                if processed_docs >= max_docs:
                    break
                    
                try:
                    # Use smaller top_k per document to limit overall chunks
                    doc_chunks = retrieve_relevant_chunks(query, [document], top_k=2)
                    all_relevant_chunks.extend(doc_chunks)
                    processed_docs += 1
                except Exception as e:
                    continue
            
            # Sort all chunks by similarity and take the top k
            sorted_chunks = sorted(all_relevant_chunks, key=lambda x: x["similarity"], reverse=True)
            top_chunks = sorted_chunks[:top_k]
            
            if not top_chunks:
                return "No relevant document chunks found."
            
            # Format response with content from most relevant chunks only
            # Use truncation to limit token usage
            response = f"Most relevant information for: '{query}'\n\n"
            
            total_content = 0
            for i, chunk in enumerate(top_chunks):
                if total_content >= MAX_CHUNK_SIZE * MAX_CHUNKS_PER_CALL:
                    break
                    
                # Truncate each chunk content to limit tokens
                content = truncate_text(chunk["content"], MAX_CHUNK_SIZE)
                total_content += len(content)
                
                # Highlight only the individual's name and financial figures
                highlighted_content = content.replace(
                    individual_name, f"**{individual_name}**"
                )
                
                response += f"--- From: {chunk['document']} ---\n"
                response += f"{highlighted_content}\n\n"
            
            # Add a brief summary of doc sources
            doc_sources = list(set(chunk["document_path"] for chunk in top_chunks))
            response += f"Sources: {len(doc_sources)} documents"
            
            return response
            
        except Exception as e:
            return f"Error retrieving relevant chunks: {str(e)}"

async def create_optimized_agent_group(
    kernel: Kernel, 
    var_individual: str
) -> AgentGroupChat:
    """Create a group of agents with document retriever, financial expert and validator."""
    
    # 1. Document Retriever Agent - Gets relevant chunks using embeddings
    document_retriever = ChatCompletionAgent(
        kernel=kernel,
        name=DOCUMENT_RETRIEVER,
        instructions=f"""
You are a financial document retrieval specialist focused on {var_individual}'s financial data.

INSTRUCTIONS:
1. Analyze the question to identify what financial information is needed
2. Use the 'retrieve_relevant_chunks' function to find relevant content
3. Focus on finding specific financial figures and values
4. Keep your response brief and direct - avoid unnecessary text

Your goal is to retrieve only the most relevant content that answers the question.
""",
        plugins=[DocumentContentPlugin()]
    )
    
    # 2. Financial Expert Agent - Generates answers based on retrieved content
    financial_expert = ChatCompletionAgent(
        kernel=kernel,
        name=FINANCIAL_EXPERT,
        instructions=f"""
You extract financial data for {var_individual} from document chunks.

INSTRUCTIONS:
1. Focus only on extracting the specific financial information requested
2. When aggregating values from multiple documents, show your calculations
3. Format currency values as "CHF 1'000'000.00"
4. Keep explanations extremely brief
5. If information isn't available, just say "Unable to determine answer from available documents"

Always format your final answer exactly according to the example format provided.
"""
    )
    
    # 3. Answer Validator Agent - Validates answers for accuracy and completeness
    answer_validator = ChatCompletionAgent(
        kernel=kernel,
        name=ANSWER_VALIDATOR,
        instructions=f"""
You validate financial data extraction for {var_individual}.

INSTRUCTIONS:
1. Verify the answer matches the format required
2. Check calculations are correct if shown
3. Ensure currency formatting follows Swiss format (e.g., "CHF 1'000'000.00")
4. Be extremely brief in your responses

If correct:
ANSWER VALIDATED: [Brief reason]

FINAL ANSWER:
[Final answer in the required format]

If incorrect:
[Brief feedback]
"""
    )

    # Function to determine which agent should take the next turn - simplified for token reduction
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Choose the next agent. Be extremely brief - just state the name:
- After {DOCUMENT_RETRIEVER} → {FINANCIAL_EXPERT}
- After {FINANCIAL_EXPERT} → {ANSWER_VALIDATOR}
- If {ANSWER_VALIDATOR} said "ANSWER VALIDATED" → stop
- Otherwise, after {ANSWER_VALIDATOR} → {FINANCIAL_EXPERT}

History:
{{{{$history}}}}
"""
    )

    # Function to determine when the conversation should end - simplified for token reduction
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt="""
If the history contains "ANSWER VALIDATED", respond with: yes
Otherwise respond with: no

History:
{{{{$history}}}}
"""
    )

    # Create a history reducer with very limited messages
    history_reducer = ChatHistoryTruncationReducer(target_count=TRUNCATE_HISTORY)

    # Create the agent group chat with reduced iterations
    return AgentGroupChat(
        agents=[document_retriever, financial_expert, answer_validator],
        selection_strategy=KernelFunctionSelectionStrategy(
            initial_agent=document_retriever,
            function=selection_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).strip(),
            agent_variable_name="agents",
            history_variable_name="history",
            history_reducer=history_reducer,
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[answer_validator],
            function=termination_function,
            kernel=kernel,
            result_parser=lambda result: str(result.value[0]).lower() == "yes",
            history_variable_name="history",
            maximum_iterations=MAX_ITERATIONS,  # Reduced iterations to save tokens
            history_reducer=history_reducer,
        ),
    )

async def process_question_with_agents(
    chat: AgentGroupChat, 
    question: Dict[str, Any], 
    var_individual: str
) -> Dict[str, str]:
    """Process a single question using the multi-agent setup with token optimization."""
    question_text = question["description"].replace("$_var_individual", var_individual)
    display_name = question["displayName"]
    example = question["example"]
    
    print(f"\n\n--- Processing: {display_name} ---")
    
    # Add the question to the chat - simplified prompt to reduce tokens
    await chat.add_chat_message(
        f"Question: {question_text}\n"
        f"Format answer like: \"{example}\""
    )

    # Process conversation for this question
    answer = None
    all_responses = []
    document_paths = []
    
    try:
        async for response in chat.invoke():
            if response is None or not response.name:
                continue
            
            # Print only agent name and truncated response to console
            print(f"\n# {response.name}:")
            print(truncate_text(response.content, 300))
            all_responses.append(response.content)

            # Extract document paths efficiently
            if response.name == DOCUMENT_RETRIEVER:
                paths = re.findall(r"processed_documents[^.]+\.json", response.content)
                document_paths.extend([p for p in paths if p not in document_paths])
            
            # When answer validator confirms the answer, extract the final answer
            if response.name == ANSWER_VALIDATOR and "ANSWER VALIDATED" in response.content:
                answer = extract_final_answer(response.content)
                break
    except Exception as e:
        print(f"Error during chat invocation: {str(e)}")
    
    # Find answer in responses if not extracted yet
    if not answer:
        for content in all_responses:
            if "FINAL ANSWER:" in content:
                answer = extract_final_answer(content)
                break
    
    # Final fallback - extract any CHF figures from the last FinancialExpert response
    if not answer:
        for content in reversed(all_responses):
            if content and FINANCIAL_EXPERT in content:
                currency_amount = extract_currency_amount(content)
                if currency_amount:
                    answer = currency_amount
                    break
        
    # If still no answer, use default fallback
    if not answer:
        answer = "Unable to determine answer from available documents."
    
    print(f"\nCompleted {display_name}: {answer}")
    
    return {
        "field_name": display_name,
        "question": question_text,
        "answer": answer,
        "sources": document_paths
    }

def extract_final_answer(content: str) -> str:
    """Extract final answer from validator response."""
    if "FINAL ANSWER:" in content:
        answer_start = content.find("FINAL ANSWER:") + len("FINAL ANSWER:")
        answer_end = content.find("\n\n", answer_start) if "\n\n" in content[answer_start:] else len(content)
        return content[answer_start:answer_end].strip()
    return None

# Cache to store retrieved document chunks for related questions
document_cache = {}

async def process_form_schema_optimized(
    schema_path: str, 
    var_individual: str = "Hans Muster"
) -> Dict[str, Any]:
    """Process form schema with extreme token optimization."""
    
    # Load necessary data
    schema = load_form_schema(schema_path)
    
    # Get the category name and questions
    category_name = list(schema.keys())[0]  # e.g., "assetsAndIncome"
    properties = schema[category_name]["properties"]
    
    # Create kernel once for all questions
    kernel = create_kernel()
    
    # Group related questions to share context and limit retrievals
    question_groups = group_related_questions(properties)
    
    # Process each group of questions
    results = {}
    
    for base_field, questions in question_groups.items():
        print(f"\n=== Processing question group: {base_field} ===")
        
        # Get first question in the group
        first_field, first_question = questions[0]
        
        # Create a fresh agent group for each question group to avoid state issues
        chat = await create_optimized_agent_group(kernel, var_individual)
        
        # Process first question in group
        result = await process_question_with_agents(chat, first_question, var_individual)
        results[first_field] = result["answer"]
        
        # Create a new chat instance for each question instead of trying to reset
        
        # Process additional questions in the group
        for field_name, question in questions[1:]:
            # Skip the first question as we already processed it
            if field_name == first_field:
                continue
            
            # Create a fresh chat for each question instead of trying to reset
            fresh_chat = await create_optimized_agent_group(kernel, var_individual)
            
            # Process with the multi-agent system
            related_result = await process_question_with_agents(fresh_chat, question, var_individual)
            results[field_name] = related_result["answer"]
    
    # Format the output according to schema
    output = {
        category_name: {}
    }
    
    # Copy results to output
    for field_name in properties:
        if field_name in results:
            output[category_name][field_name] = results[field_name]
        else:
            output[category_name][field_name] = "Unable to determine answer from available documents."
    
    return output

async def main():
    # Set the individual name to look for in documents
    var_individual = "Hans Muster"
    
    # Path to the schema file
    schema_path = "schema/schema_assetsAndIncome.json"
    
    print(f"Processing form for individual: {var_individual}")
    print(f"Using schema: {schema_path}")
    print("-" * 50)
    
    # Track token usage and execution time
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