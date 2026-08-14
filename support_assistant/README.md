Zepto Support Assistant
Introduction

Zepto Support Assistant is a simple RAG based application that can answer questions related to Zepto policies.

RAG stands for Retrieval Augmented Generation. In this project, the application first searches the available Zepto policy documents and then uses the relevant information to provide an answer.

The application is developed using Python, FastAPI, LangGraph, ChromaDB, Sentence Transformers, and Pydantic.

The application uses a mock mode by default. This means that a Groq API key is not required to run the basic version of the project.

Features

The main features of the project are:

Loads 8 Zepto policy documents from the docs directory.
Converts each policy document into a searchable chunk.
Creates embeddings using the Sentence Transformer model.
Stores the embeddings in ChromaDB.
Uses LangGraph to classify and route user questions.
Retrieves the top 3 relevant policy documents for policy related questions.
Uses MOCK_LLM mode by default.
Can optionally use a Groq LLM.
Uses Pydantic to validate responses.
Retries invalid responses up to 3 times when using the real LLM.
Provides a FastAPI API endpoint at /ask.
Provides a health check endpoint at /.
Technologies Used

The following technologies are used in this project:

Technology Purpose
Python Main programming language
FastAPI Creating the REST API
Pydantic Validating request and response data
LangGraph Managing the application workflow
ChromaDB Storing and searching embeddings
Sentence Transformers Creating text embeddings
Groq Optional real LLM
LangChain Groq Connecting to Groq
Uvicorn Running the FastAPI application
Basic Architecture

The application works in the following steps:

The user sends a question to the /ask endpoint.
The application checks whether the question is related to a Zepto policy.
If it is a policy question, the application searches ChromaDB.
The top 3 relevant policy chunks are retrieved.
The application generates an answer.
If it is a general question, the application provides a direct response.
Pydantic validates the final response.
The application returns the response as JSON.

The main workflow can be described as:

User Question
|
v
Intent Classification
|
+-----------------------+
| |
v v
Policy Question General Question
| |
v v
Search ChromaDB Direct Answer
|
v
Retrieve Top 3 Results
|
v
Generate Answer
|
v
Pydantic Validation
|
v
JSON Response
Project Structure

The expected project structure is:

support_assistant/
|
|-- main.py
|-- README.md
|-- requirements.txt
|
|-- docs/
| |-- doc_01.txt
| |-- doc_02.txt
| |-- doc_03.txt
| |-- doc_04.txt
| |-- doc_05.txt
| |-- doc_06.txt
| |-- doc_07.txt
| |-- doc_08.txt
|
|-- chroma_db/
| |-- ...

The chroma_db directory is created automatically when the application starts.

Prerequisites

Before running the application, the following are required:

Python 3.10 or later.
pip.
Internet access during the first run to download the Sentence Transformer model.
Eight Zepto policy files.

A Groq API key is not required when using MOCK_LLM mode.

Installation
Step 1: Create the Project Directory

Create a directory for the project:

support_assistant/

Place the eight policy documents inside the docs directory:

docs/doc_01.txt
docs/doc_02.txt
docs/doc_03.txt
docs/doc_04.txt
docs/doc_05.txt
docs/doc_06.txt
docs/doc_07.txt
docs/doc_08.txt
Step 2: Create a Virtual Environment

For Windows:

python -m venv .venv
.venv\Scripts\activate

For Linux or macOS:

python3 -m venv .venv
source .venv/bin/activate
Step 3: Install Dependencies

The requirements.txt file can contain:

fastapi
uvicorn
pydantic
chromadb
sentence-transformers
langgraph
langchain-groq

Install the dependencies using:

pip install -r requirements.txt
Running the Application

The application uses MOCK_LLM mode by default.

Start the application using:

uvicorn main:app --reload

The application will normally run at:

http://127.0.0.1:8000

The FastAPI documentation can be opened at:

http://127.0.0.1:8000/docs

The /docs page provides an interactive interface for testing the API.

MOCK_LLM Mode

MOCK_LLM is used to run the application without an external LLM API.

Mock mode is enabled when MOCK_LLM is not set or when it has a value other than 0.

For example:

uvicorn main:app --reload

or:

MOCK_LLM=1 uvicorn main:app --reload

In mock mode:

Intent classification uses predefined keywords.
Embeddings are still generated.
ChromaDB retrieval is still performed.
Policy answers are generated using a predefined response based on the retrieved document.
General questions receive a fixed response.
No external LLM API is called.
Policy Keywords

The mock classifier checks for the following keywords:

delivery
return
refund
membership
tracking
cancel
gift card
support hours

If one of these keywords is found in the question, the application classifies it as:

policy_question

If none of the keywords are found, it is classified as:

general_question
Optional Real LLM Mode

The application can also use a Groq hosted LLM.

To enable real LLM mode, set:

MOCK_LLM=0

and provide a Groq API key using:

GROQ_API_KEY

For Linux or macOS:

export MOCK_LLM=0
export GROQ_API_KEY="your-api-key"
uvicorn main:app --reload

For Windows PowerShell:

$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your-api-key"
uvicorn main:app --reload

The default Groq model is:

llama-3.1-8b-instant

The model can be changed using:

GROQ_MODEL

For example:

export GROQ_MODEL="your-model-name"

API keys should be stored in environment variables or a secure secret manager. They should not be added to source code or uploaded to a public repository.

Document Processing

When main.py starts, the application performs the following steps:

Step 1: Load the eight policy documents
Step 2: Create one chunk from each document
Step 3: Generate embeddings
Step 4: Store the embeddings in ChromaDB
Step 5: Start the FastAPI application

The embedding model used in this project is:

sentence-transformers/all-MiniLM-L6-v2

The ChromaDB collection name is:

zepto_policies

The ChromaDB database is stored in:

chroma_db/

The application uses cosine distance to compare the similarity between the user question and the policy documents.

The application uses upsert when storing documents. This helps prevent the same document ID from being intentionally inserted multiple times when the application is restarted.

API Reference
GET /

The GET / endpoint is used as a health check.

Request:

GET /

Example response:

{
"message": "Zepto Support Assistant is running.",
"mock_llm": true,
"collection": "zepto_policies"
}
POST /ask

The POST /ask endpoint is used to ask a question.

Request:

POST /ask
Content-Type: application/json

Example request:

{
"query": "How much is delivery below INR 149?"
}
Example Using cURL
curl -X POST "http://127.0.0.1:8000/ask" \
-H "Content-Type: application/json" \
-d '{"query":"How much is delivery below INR 149?"}'
Response Format

A successful response has the following format:

{
"answer": "Based on the retrieved context: ...",
"sources": [
"doc_01",
"doc_03",
"doc_05"
],
"confidence": 1.0
}
Response Schema

The response contains three fields:

Field Type Description
answer string The generated answer
sources array of strings IDs of the retrieved documents
confidence float Confidence value between 0.0 and 1.0

Pydantic checks that the confidence value is between 0.0 and 1.0.

Example Questions
Example 1: Delivery Question

Request:

{
"query": "What are the delivery charges?"
}

Since the question contains the word delivery, it is classified as a policy question.

The application follows this process:

classify_intent
|
v
policy_question
|
v
retrieve_and_answer
|
v
ChromaDB
|
v
Top 3 Policy Chunks
|
v
Answer
Example 2: Refund Question

Request:

{
"query": "How can I get a refund?"
}

The word refund is a policy keyword, so the application sends the question to the retrieval process.

Example 3: General Question

Request:

{
"query": "What is the capital of India?"
}

This question does not contain any configured Zepto policy keyword.

Therefore, it is classified as:

general_question

In MOCK_LLM mode, the response is:

{
"answer": "I can only answer questions about Zepto policies right now.",
"sources": [],
"confidence": 1.0
}
LangGraph Workflow

The application has three main LangGraph nodes.

1. classify_intent

This node determines whether the question is a policy question or a general question.

The two possible results are:

policy_question

and:

general_question

In mock mode, keyword matching is used for classification.

2. retrieve_and_answer

This node is used for policy questions.

It performs the following tasks:

Converts the user question into an embedding.
Searches ChromaDB.
Retrieves the top 3 relevant chunks.
Creates the context for the answer.
Generates the answer.
Returns the source IDs and confidence.

In mock mode, the answer is based on the first retrieved document.

In real LLM mode, the retrieved information is provided to the LLM.

3. direct_answer

This node is used for general questions.

In mock mode, it returns:

I can only answer questions about Zepto policies right now.

In real LLM mode, the configured Groq model is called without searching the policy documents.

Real LLM Validation

When real LLM mode is enabled, the model is asked to return a JSON response.

The expected format is:

{
"answer": "string",
"sources": ["string"],
"confidence": 0.0
}

The response is checked using the AnswerResponse Pydantic model.

If the response is invalid, the application tries again.

The process can be described as:

Attempt 1
|
v
Validation
|
v
If invalid, send corrective prompt
|
v
Attempt 2
|
v
Validation
|
v
If invalid, send corrective prompt
|
v
Attempt 3

If all three attempts fail, the application returns:

{
"answer": "ERROR: The real LLM response could not be validated after three attempts.",
"sources": [],
"confidence": 0.0
}
Retrieval Process

For every policy question, the following process takes place:

User Question
|
v
Sentence Transformer
|
v
Embedding Vector
|
v
ChromaDB Search
|
v
Top 3 Results

A retrieved result can contain information such as:

{
"id": "doc_01",
"text": "...",
"source": "doc_01.txt",
"distance": 0.123
}

The first result is considered the closest match.

Configuration
Variable Default Description
MOCK_LLM 1 Enables mock mode unless set to 0
GROQ_API_KEY None Required only for real LLM mode
GROQ_MODEL llama-3.1-8b-instant Groq model used in real LLM mode

The important behavior is:

MOCK_LLM is not set
Mock mode

MOCK_LLM=1
Mock mode

MOCK_LLM=0
Real LLM mode
Error Handling

The application handles several common errors.

Missing Policy Document

If one of the required policy documents is missing, the application raises a FileNotFoundError.

The required files are:

docs/doc_01.txt
docs/doc_02.txt
docs/doc_03.txt
docs/doc_04.txt
docs/doc_05.txt
docs/doc_06.txt
docs/doc_07.txt
docs/doc_08.txt
Missing Groq API Key

If real LLM mode is enabled but the Groq API key is not provided, the application raises a configuration error.

Invalid LLM Response

If the real LLM returns invalid JSON or does not follow the required structure, the application retries the request up to three times.

Empty Question

If the user sends an empty question:

{
"query": ""
}

the application returns:

{
"answer": "Please provide a question.",
"sources": [],
"confidence": 0.0
}
Design Principles
Local Embeddings

The project creates embeddings locally using Sentence Transformers.

This means that an external embedding API is not required.

Persistent Vector Database

ChromaDB uses a persistent directory so that the stored data remains available after the application is restarted.

Simple Default Mode

The default mock mode makes the project easier to run and test.

A Groq API key is not required for basic testing.

Retrieval Works in Mock Mode

An important part of this project is that MOCK_LLM does not disable ChromaDB retrieval.

When MOCK_LLM is enabled:

Embeddings are created normally.
ChromaDB search is performed normally.
Intent classification uses simple keywords.
Answer generation uses the predefined mock response.
Structured Output

Pydantic is used to validate the response returned by the application.

This helps ensure that the response contains the required fields and that the confidence value is valid.

Testing

After starting the application, open:

http://127.0.0.1:8000/docs

Use the POST /ask endpoint to test the application.

Test 1: Delivery

Request:

{
"query": "What are the delivery charges?"
}

Expected classification:

policy_question

Expected workflow:

policy_question
|
v
retrieve_and_answer
Test 2: Refund

Request:

{
"query": "How do I get a refund?"
}

Expected classification:

policy_question

Expected workflow:

policy_question
|
v
retrieve_and_answer
Test 3: Membership

Request:

{
"query": "What does membership include?"
}

Expected classification:

policy_question

Expected workflow:

policy_question
|
v
retrieve_and_answer
Test 4: General Question

Request:

{
"query": "Tell me a joke."
}

Expected classification:

general_question

Expected workflow:

general_question
|
v
direct_answer
Test 5: Empty Input

Request:

{
"query": ""
}

Expected answer:

Please provide a question.
Startup Sequence

When the application starts, messages similar to the following may appear:

Loading local embedding model...
Embedding model loaded.
Loading Zepto documents...
Loaded 8 documents.
Created 8 chunks.
Creating embeddings...
Created 8 embeddings.
ChromaDB collection 'zepto_policies' contains 8 records.

The exact messages may be different depending on the installed library versions and the current ChromaDB data.

Limitations

This is a beginner level implementation, so there are some limitations.

Each document is represented by one chunk.
The intent classifier uses simple keyword matching.
The mock answer is based on the closest retrieved document.
The application retrieves up to three results.
There is no specific similarity threshold.
The real LLM depends on the configured Groq model.
The application currently provides one main question answering endpoint.
The application does not include authentication.
The application does not include advanced monitoring or logging.

These limitations are acceptable for a learning project and demonstration.

Future Improvements

The project can be improved in the future by adding:

Better intent classification.
Different chunk sizes and overlapping chunks.
Similarity thresholds for retrieval.
Better source information in the answers.
Conversation history.
User authentication.
Rate limiting.
Automated unit tests.
Integration tests.
Better application logging.
Docker deployment.
Production database configuration.
RAG evaluation datasets.
LLM monitoring and tracing.
Better handling when no relevant policy is found.
License

The appropriate project or company license can be added here.

Conclusion

The Zepto Support Assistant is a simple RAG based application created to demonstrate how different AI and software technologies can work together.

The basic process is:

Zepto Policy Documents
|
v
Sentence Transformer
|
v
Embeddings
|
v
ChromaDB
|
v
User Question
|
v
LangGraph Intent Classification
|
+-------------------------+
| |
v v
Policy Question General Question
| |
v v
Retrieve Top 3 Direct Answer
|
v
Mock or Real LLM
|
v
Pydantic Validation
|
v
FastAPI JSON Response

This project helped demonstrate the basic concepts of RAG, vector search, embeddings, LangGraph workflow management, structured responses, and FastAPI.

The use of MOCK_LLM mode also makes the project easier for a beginner to run and understand without requiring an external LLM API key.
