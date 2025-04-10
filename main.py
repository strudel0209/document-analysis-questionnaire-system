import asyncio
import json
import os
from typing import Dict, List, Any

from semantic_kernel import Kernel
from semantic_kernel.agents import AgentGroupChat, ChatCompletionAgent
from semantic_kernel.agents.strategies import (
    KernelFunctionSelectionStrategy,
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents import ChatHistoryTruncationReducer
from semantic_kernel.functions import KernelFunctionFromPrompt
from typing import TypedDict, Annotated, List, Optional
from semantic_kernel.functions import kernel_function

# Define agent names
DOCUMENT_RETRIEVER = "DocumentRetriever"
ANSWER_GENERATOR = "AnswerGenerator"
ANSWER_VALIDATOR = "AnswerValidator"

class DocumentContentPlugin:
    def _load_document_content(self, path: str) -> Optional[str]:
        """Loads document content from a file if it exists."""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as file:
                return file.read()
        return None

    @kernel_function
    async def retrieve_documents(
        self,
        document_list: Annotated[List[str], "List of relevant document paths"]
    ) -> str:
        """
        Fetches content of the documents selected by the Document Retriever Agent.
        Expects a list of document paths and returns a structured response with content.
        """
        if not document_list:
            return "No relevant documents were found."

        content_message = "Here are the contents of the relevant documents:\n\n"

        for doc_path in document_list:
            # Convert `_processed.json` to `_content.md`
            content_path = doc_path.replace("_processed.json", "_content.md")
            content = self._load_document_content(content_path)

            if content:
                content_message += f"--- DOCUMENT: {os.path.basename(doc_path)} ---\n{content}\n\n"
            else:
                content_message += f"--- DOCUMENT: {os.path.basename(doc_path)} ---\n[Error: Unable to load content]\n\n"

        return content_message

def create_kernel() -> Kernel:
    """Creates a Kernel instance with an Azure OpenAI ChatCompletion service."""
    kernel = Kernel()
    kernel.add_service(service=AzureChatCompletion())
    return kernel

def load_document_index(index_path: str = "processed_documents/document_index.json") -> List[Dict[str, Any]]:
    """Load the document index containing metadata for all processed documents."""
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_document_content(content_path: str) -> str:
    """Load the markdown content of a processed document."""
    try:
        with open(content_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading document content from {content_path}: {e}")
        return f"[Error loading content: {str(e)}]"

def load_form_schema(schema_path: str = "schema/schema_assetsAndIncome.json") -> Dict[str, Any]:
    """Load the form schema containing questions to answer."""
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

async def create_qa_agents(kernel: Kernel, document_index: List[Dict[str, Any]], var_individual: str) -> AgentGroupChat:
    """Create the agent group chat with document retriever, answer generator, and validator agents."""
    
    # Create a summary of all documents for the document retriever
    doc_summaries = "\n\n".join([
        f"Document {i+1}:\n" +
        f"- Title: {doc['metadata']['title']}\n" +
        f"- Type: {doc['metadata']['document_type']}\n" +
        f"- Description: {doc['metadata']['description']}\n" +
        f"- Summary: {doc['metadata']['summary']}\n" +
        f"- Key entities: {', '.join(doc['metadata']['key_entities'])}\n" + 
        f"- Categories: {', '.join(doc['metadata']['categories'])}\n" +
        f"- Path: {doc['processed_path']}"
        for i, doc in enumerate(document_index)
    ])
    
    # 1. Document Retriever Agent - Selects relevant documents for a question
    document_retriever = ChatCompletionAgent(
        kernel=kernel,
        name=DOCUMENT_RETRIEVER,
    instructions=f"""
You are a document retrieval specialist focused on financial information. Your job is to analyze questions and identify the most relevant documents that contain the information needed to provide precise answers.

Available Documents:
{doc_summaries}

INSTRUCTIONS:
1. Analyze the question to identify specific financial data being requested (assets, income, values, etc.)
2. Select documents that are MOST likely to contain this specific information, considering:
   - Document type (reports and statements are often more reliable for financial figures)
   - Relevance to {var_individual} specifically
   - Time period mentioned (prioritize recent documents)
   - Financial category alignment (bankable assets, real estate, income sources, etc.)
3. Be selective - quality over quantity. Usually 1-3 highly relevant documents are better than many tangential ones
4. Always provide the FULL PATH to each document exactly as listed in the metadata (e.g., "processed_documents\\bsp_Hans_Muster\\...")
5. For each document, provide the full document content exactly as it is retrieved without any modifications including formatting or structure

The questions are about financial information for {var_individual}.
Remember that you are only selecting documents - do not attempt to answer the question yourself.

RESPONSE FORMAT:
---
Selected documents:

1. [FULL DOCUMENT PATH]
Relevance: [Specific explanation of how this document answers the question]
Expected Information: [What specific data you expect to find in this document]
Content:
[DOCUMENT CONTENT]

2. [FULL DOCUMENT PATH]
Relevance: [Specific explanation of how this document answers the question]
Expected Information: [What specific data you expect to find in this document]
Content:
[DOCUMENT CONTENT]
---
""",
    plugins=[DocumentContentPlugin()]
    )
    
    # 2. Answer Generator Agent - Generates answers based on document content
    answer_generator = ChatCompletionAgent(
        kernel=kernel,
        name=ANSWER_GENERATOR,
        instructions=f"""
You are a financial data extraction specialist. Your job is to answer questions by carefully analyzing document content.

INSTRUCTIONS:
1. Review the document content provided to you
2. Extract specific financial information requested in the question
3. Format currency values consistently (e.g., "CHF 1'000'000.00")
4. If exact values aren't available, provide the best estimate based on available information
5. ALWAYS cite the specific document name and section where you found information
6. If information is not available in the documents, respond with "Information not available in provided documents"
7. MATCH THE FORMAT shown in the example response exactly - pay special attention to currency format, notation, and structure

Questions are about financial information for {var_individual}.
Be precise and accurate - these answers will be used in an official form.

RESPONSE FORMAT:
- Structure your answer EXACTLY as shown in the example response
- Include all necessary details while matching the example's level of detail
- Format currency values consistently with the example (e.g., "CHF 5'000'000.00")
- Include sources/citations at the appropriate location based on the example

If the validator provides feedback, be sure to address ALL points in your revised answer.
"""
    )
    
    # 3. Answer Validator Agent - Validates answers for accuracy and completeness
    answer_validator = ChatCompletionAgent(
        kernel=kernel,
        name=ANSWER_VALIDATOR,
        instructions="""
You are an answer validation specialist for financial forms. Your task is to verify answers meet quality standards and provide the final validated answer.

INSTRUCTIONS:
1. Evaluate if the answer directly addresses the question
2. Check if the answer includes specific financial values when required
3. Verify the answer follows EXACTLY the format shown in the example response
4. Ensure currency formatting, notation, and structure match the example precisely
5. Ensure the answer cites sources from the documents

RESPONSE FORMAT:
- If the answer needs improvement: Provide SPECIFIC feedback on what needs to be fixed, but fix format issues yourself

- If the answer is complete and accurate: Format your response EXACTLY as follows:
  ANSWER VALIDATED: [Brief reason why the answer is good]
  
  FINAL ANSWER:
  [The complete final answer, formatted EXACTLY according to the example response]

IMPORTANT:
- NEVER use "ANSWER VALIDATED:" unless the answer fully meets all quality standards AND follows the example format
- Always check currency notation, digit grouping, decimal places, and overall structure against the example
- When validating, always include both the validation statement AND the final answer
- The final answer should be formatted precisely according to the example response

Always maintain this exact format to ensure the conversation flow works correctly.
"""
    )

    # Function to determine which agent should take the next turn
    selection_function = KernelFunctionFromPrompt(
        function_name="selection",
        prompt=f"""
Determine which participant takes the next turn in a conversation based on the the most recent participant.
State only the name of the participant to take the next turn.
No participant should take more than one turn in a row.

Choose only from these participants:
- {DOCUMENT_RETRIEVER}
- {ANSWER_GENERATOR}
- {ANSWER_VALIDATOR}

Always follow these rules when selecting the next participant:
- After {DOCUMENT_RETRIEVER}, it is {ANSWER_GENERATOR}'s turn.
- After {ANSWER_GENERATOR} replies, it is {ANSWER_VALIDATOR}'s turn.
- If response from {ANSWER_VALIDATOR} does NOT contain the exact phrase "ANSWER VALIDATED", it is {ANSWER_GENERATOR}'s turn.
- If response from {ANSWER_VALIDATOR} contains the exact phrase "ANSWER VALIDATED", the conversation is complete.

History:
{{{{$history}}}}
"""
    )

    # Function to determine when the conversation should end
    termination_function = KernelFunctionFromPrompt(
        function_name="termination",
        prompt="""
Examine the history and determine whether the answer has been validated.
If the answer has been validated (contains "ANSWER VALIDATED"), respond with a single word: yes.

History:
{{{{$history}}}}
"""
    )

    # Create a history reducer to keep context manageable
    history_reducer = ChatHistoryTruncationReducer(target_count=8)

    # Create the agent group chat
    return AgentGroupChat(
        agents=[document_retriever, answer_generator, answer_validator],
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
            maximum_iterations=10,
            history_reducer=history_reducer,
        ),
    )

async def process_question(chat: AgentGroupChat, question: Dict[str, Any], var_individual: str) -> Dict[str, str]:
    """Process a single question and return the answer."""
    # Prepare the question with variable substitution
    question_text = question["description"].replace("$_var_individual", var_individual)
    display_name = question["displayName"]
    example = question["example"]
    
    print(f"\n\n--- Processing: {display_name} ---")
    print(f"Question: {question_text}")
    
    # Add the question to the chat
    await chat.add_chat_message(
        f"Question: {question_text}\n\n"
        f"FORMAT REQUIREMENT: Your answer must follow exactly this example format: \"{example}\"\n"
        f"Pay special attention to currency notation, spacing, and structure."
    )

    # Process conversation for this question
    answer = None
    document_paths = []
    
    async for response in chat.invoke():
        if response is None or not response.name:
            continue
        
        print(f"\n# {response.name}:")
        #print(response.content[:300] + "..." if len(response.content) > 300 else response.content)
        print(response.content)

        # Extract document paths from the DocumentRetriever's response
        if response.name == DOCUMENT_RETRIEVER:
            # Look for document paths in the response
            lines = response.content.split('\n')
            for line in lines:
                if "processed_documents" in line:
                    # Find path start and end positions
                    path_start = line.find("processed_documents")
                    path_end = line.find(".json", path_start)
                    if path_start >= 0 and path_end >= 0:
                        doc_path = line[path_start:path_end + 5]  # Include the .json extension
                        if doc_path not in document_paths:
                            document_paths.append(doc_path)
                            print(f"Found document path: {doc_path}")
                
        # When answer validator confirms the answer, extract the final answer
        if response.name == ANSWER_VALIDATOR and "ANSWER VALIDATED" in response.content:
            # Extract the final answer from the validator's response
            content = response.content
            if "FINAL ANSWER:" in content:
                answer_start = content.find("FINAL ANSWER:") + len("FINAL ANSWER:")
                answer = content[answer_start:].strip()
    
    # Check if we got a valid answer
    if not answer:
        answer = "Unable to determine answer from available documents."

    # Print found document paths
    if document_paths:
        print(f"Document paths used: {document_paths}")
    else:
        print("No document paths were extracted")

    return {
        "field_name": display_name,
        "question": question_text,
        "answer": answer,
        "sources": document_paths
    }

async def process_form_schema(schema_path: str, var_individual: str = "Hans Muster") -> Dict[str, Any]:
    """Process all questions in the form schema and generate answers."""
    
    # Load necessary data
    schema = load_form_schema(schema_path)
    document_index = load_document_index()
    
    # Get the category name and questions
    category_name = list(schema.keys())[0]  # e.g., "assetsAndIncome"
    properties = schema[category_name]["properties"]
    
    # Create kernel and agents
    kernel = create_kernel()
    chat = await create_qa_agents(kernel, document_index, var_individual)
    
    # Process each question
    results = {}
    
    for field_name, question in properties.items():
        try:
            result = await process_question(chat, question, var_individual)
            results[field_name] = result["answer"]
            print(f"\nCompleted {field_name}: {result['answer']}")
            # Evaluate if completion is met and reset
            if chat.is_complete:
                # Reset completion state to continue use
                chat.is_complete = False
                await chat.reset()
        except Exception as e:
            print(f"Error processing {field_name}: {e}")
            results[field_name] = f"Error: {str(e)}"
    
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
    
    # Process the schema and generate answers
    results = await process_form_schema(schema_path, var_individual)
    
    # Save results to file
    output_path = f"{var_individual.replace(' ', '_')}_form_answers.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print("\nProcessing complete!")
    print(f"Results saved to: {output_path}")
    
    # Print a summary of results
    print("\nSummary of answers:")
    for field, answer in results["assetsAndIncome"].items():
        print(f"{field}: {answer[:50]}..." if len(answer) > 50 else answer)

if __name__ == "__main__":
    asyncio.run(main())