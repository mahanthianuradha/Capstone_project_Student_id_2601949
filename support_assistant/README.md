Zepto Support Assistant

A RAG-based Zepto policy support assistant built with FastAPI, LangGraph, ChromaDB, Sentence Transformers, and Pydantic.

The application loads Zepto policy documents, creates local embeddings, stores them in ChromaDB, routes incoming questions using LangGraph, retrieves relevant policy content when required, and returns a validated JSON response.

By default, the application runs in deterministic MOCK_LLM mode, so an external LLM API key is not required.

Features
📄 Loads 8 Zepto policy documents from the docs/ directory.
✂️ Converts each policy document into a searchable chunk.
🧠 Generates embeddings using:
sentence-transformers/all-MiniLM-L6-v2
🗄️ Stores embeddings in persistent ChromaDB.
🔀 Uses LangGraph for intent classification and routing.
🔎 Retrieves the top 3 most relevant policy chunks for policy questions.
🤖 Supports deterministic MOCK_LLM mode by default.
🚀 Optionally supports a real Groq LLM.
✅ Uses Pydantic to validate LLM responses.
🔁 Retries invalid real-LLM responses up to 3 attempts.
🌐 Provides a FastAPI /ask endpoint.
❤️ Provides a simple health-check endpoint at /.
Architecture
┌──────────────────┐
│ POST /ask │
│ User Question │
└────────┬─────────┘
│
▼
┌─────────────────────────┐
│ classify_intent │
│ │
│ policy_question ? │
│ general_question ? │
└───────────┬─────────────┘
│
┌──────────────┴──────────────┐
│ │
▼ ▼
┌─────────────────────┐ ┌──────────────────┐
│ retrieve_and_answer │ │ direct_answer │
│ │ │ │
│ Embed query │ │ No retrieval │
│ Search ChromaDB │ │ │
│ Top 3 chunks │ │ General response │
│ Generate answer │ │ │
└──────────┬──────────┘ └────────┬─────────┘
│ │
└────────────┬─────────────┘
▼
┌────────────────────────┐
│ Pydantic Validation │
│ │
│ answer │
│ sources │
│ confidence │
└───────────┬────────────┘
│
▼
JSON API Response
Project Structure

The expected project structure is:

support_assistant/
│
├── main.py
├── README.md
├── requirements.txt
│
├── docs/
│ ├── doc_01.txt
│ ├── doc_02.txt
│ ├── doc_03.txt
│ ├── doc_04.txt
│ ├── doc_05.txt
│ ├── doc_06.txt
│ ├── doc_07.txt
│ └── doc_08.txt
│
└── chroma_db/
└── ...

chroma_db/ is created automatically when the application starts.

Prerequisites
Python 3.10+ recommended
pip
Internet access on the first run to download the Sentence Transformer model
The eight required policy files

A Groq API key is not required when using the default mock mode.

Installation

1. Clone or copy the project

Place the application in a directory such as:

support_assistant/

Ensure the eight policy files are present:

docs/doc_01.txt
docs/doc_02.txt
...
docs/doc_08.txt 2. Create a virtual environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux / macOS
python3 -m venv .venv
source .venv/bin/activate 3. Install dependencies

Example requirements.txt:

fastapi
uvicorn
pydantic
chromadb
sentence-transformers
langgraph
langchain-groq

Install them with:

pip install -r requirements.txt
Running the Application

The application uses MOCK_LLM mode by default.

Start the FastAPI server with:

uvicorn main:app --reload

The API will normally be available at:

http://127.0.0.1:8000

FastAPI's interactive API documentation will be available at:

http://127.0.0.1:8000/docs
MOCK_LLM Mode

Mock mode is enabled when:

MOCK_LLM

is missing or has any value other than 0.

For example:

uvicorn main:app --reload

or:

MOCK_LLM=1 uvicorn main:app --reload

In this mode:

Intent classification uses predefined keywords.
Embeddings are still created normally.
ChromaDB retrieval is still performed normally.
Policy answers use a deterministic canned response based on the closest retrieved document.
General questions receive a fixed response.
No external LLM API call is made.
Policy keywords

The mock classifier recognizes these keywords:

delivery
return
refund
membership
tracking
cancel
gift card
support hours

If one of these keywords occurs in the user's question, it is classified as:

policy_question

Otherwise it is classified as:

general_question
Optional Real LLM Mode

The application can optionally use a Groq-hosted LLM.

Set:

MOCK_LLM=0

and provide:

GROQ_API_KEY

For example, on Linux/macOS:

export MOCK_LLM=0
export GROQ_API_KEY="your-api-key"
uvicorn main:app --reload

On Windows PowerShell:

$env:MOCK_LLM="0"
$env:GROQ_API_KEY="your-api-key"
uvicorn main:app --reload

The default Groq model is:

llama-3.1-8b-instant

It can be changed using:

GROQ_MODEL

Example:

export GROQ_MODEL="your-model-name"

Keep API keys in environment variables or a secure secret manager. Do not commit them to source control.

Document Ingestion and Vector Database

When main.py starts, the application automatically performs the following steps:

1. Load doc_01.txt through doc_08.txt
   ↓
2. Create one chunk per document
   ↓
3. Generate Sentence Transformer embeddings
   ↓
4. Store embeddings in ChromaDB
   ↓
5. Start the FastAPI application

The embedding model is:

sentence-transformers/all-MiniLM-L6-v2

The ChromaDB collection is:

zepto_policies

The database is persisted in:

chroma_db/

The application uses cosine distance for vector similarity.

Because upsert() is used, restarting the application does not intentionally create duplicate records for the same document IDs.

API Reference
GET /

Health-check endpoint.

Request
GET /
Example response
{
"message": "Zepto Support Assistant is running.",
"mock_llm": true,
"collection": "zepto_policies"
}
POST /ask

Ask the support assistant a question.

Request
POST /ask
Content-Type: application/json
Request body
{
"query": "How much is delivery below INR 149?"
}
Example using cURL
curl -X POST "http://127.0.0.1:8000/ask" \
 -H "Content-Type: application/json" \
 -d '{"query":"How much is delivery below INR 149?"}'
Response format
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

Every successful response follows this structure:

{
"answer": "string",
"sources": ["string"],
"confidence": 0.0
}
Fields
Field Type Description
answer string Generated response
sources array of strings IDs of retrieved documents/chunks
confidence float Confidence value between 0.0 and 1.0

Pydantic enforces:

confidence >= 0.0
confidence <= 1.0
Example Questions
Policy question
{
"query": "What are the delivery charges?"
}

Because the question contains delivery, it is classified as a policy question.

Flow:

classify_intent
↓
policy_question
↓
retrieve_and_answer
↓
ChromaDB
↓
Top 3 policy chunks
↓
Answer
Another policy question
{
"query": "How can I get a refund?"
}

refund is a policy keyword, so the request is routed to retrieval.

General question
{
"query": "What is the capital of India?"
}

This does not contain any configured policy keyword.

It is therefore classified as:

general_question

In MOCK_LLM mode, the response is:

{
"answer": "I can only answer questions about Zepto policies right now.",
"sources": [],
"confidence": 1.0
}
LangGraph Workflow

The application defines three main graph nodes.

1. classify_intent

Determines whether the request is:

policy_question

or:

general_question

In mock mode, this is performed using keyword matching.

2. retrieve_and_answer

Used for policy questions.

It:

Embeds the user query.
Searches ChromaDB.
Retrieves the top 3 chunks.
Builds the context.
Generates an answer.
Returns source IDs and confidence.

In mock mode, the answer is based on the first retrieved chunk.

In real LLM mode, the retrieved context is passed to the LLM through the structured prompt.

3. direct_answer

Used for general questions.

In mock mode, it returns:

I can only answer questions about Zepto policies right now.

In real LLM mode, it calls the configured Groq model without retrieving policy documents.

Real LLM Validation

When real LLM mode is enabled, the model is instructed to return JSON:

{
"answer": "string",
"sources": ["string"],
"confidence": 0.0
}

The response is validated using the AnswerResponse Pydantic model.

If validation fails:

Attempt 1
↓
Validation failure
↓
Corrective prompt
↓
Attempt 2
↓
Validation failure
↓
Corrective prompt
↓
Attempt 3

If all three attempts fail, the application returns:

{
"answer": "ERROR: The real LLM response could not be validated after three attempts.",
"sources": [],
"confidence": 0.0
}
Retrieval Details

For every policy question, the application:

User query
↓
Sentence Transformer embedding
↓
Normalized embedding vector
↓
ChromaDB cosine similarity search
↓
Top 3 results

Each retrieved result contains:

{
"id": "doc_01",
"text": "...",
"source": "doc_01.txt",
"distance": 0.123
}

The first result is considered the closest match.

Configuration
Variable Default Description
MOCK_LLM 1 Enables mock mode unless explicitly set to 0
GROQ_API_KEY None Required only for real LLM mode
GROQ_MODEL llama-3.1-8b-instant Groq model used in real mode

Important behavior:

MOCK_LLM unset → Mock mode
MOCK_LLM=1 → Mock mode
MOCK_LLM=0 → Real LLM mode
Error Handling

The application handles several common errors.

Missing policy document

If any of the required files are missing:

docs/doc_01.txt
...
docs/doc_08.txt

the application raises a FileNotFoundError.

Missing Groq API key

If:

MOCK_LLM=0

but GROQ_API_KEY is not provided, the application raises an appropriate configuration error.

Invalid LLM JSON

Invalid JSON or schema validation failures trigger the real-LLM retry mechanism.

Empty question

An empty request such as:

{
"query": ""
}

returns:

{
"answer": "Please provide a question.",
"sources": [],
"confidence": 0.0
}
Technology Stack
Technology Purpose
Python Application language
FastAPI REST API
Pydantic Request/response validation
LangGraph Workflow orchestration
ChromaDB Vector database
Sentence Transformers Local embeddings
Groq / LangChain Groq Optional real LLM
Uvicorn ASGI server
Design Principles
Local embeddings

Embeddings are generated locally using Sentence Transformers. An external embedding API is not required.

Persistent vector storage

ChromaDB uses a persistent directory so the vector database survives application restarts.

Deterministic default behavior

The default mock implementation makes the application easy to run and test without an LLM API key.

Retrieval is independent of MOCK_LLM

An important design detail is that:

MOCK_LLM = 1

does not disable embeddings or ChromaDB retrieval.

Retrieval remains real; only answer generation and intent classification use deterministic mock logic.

Structured output

Pydantic provides an additional validation layer for responses returned by the optional LLM.

Testing

After starting the application, open:

/docs

through the FastAPI Swagger UI and use the POST /ask endpoint.

Suggested test cases:

Test 1 — Delivery
{
"query": "What are the delivery charges?"
}

Expected routing:

policy_question
→ retrieve_and_answer
Test 2 — Refund
{
"query": "How do I get a refund?"
}

Expected routing:

policy_question
→ retrieve_and_answer
Test 3 — Membership
{
"query": "What does membership include?"
}

Expected routing:

policy_question
→ retrieve_and_answer
Test 4 — General question
{
"query": "Tell me a joke."
}

Expected routing:

general_question
→ direct_answer
Test 5 — Empty input
{
"query": ""
}

Expected answer:

Please provide a question.
Startup Sequence

On startup, the console will show messages similar to:

Loading local embedding model...
Embedding model loaded.
Loading Zepto documents...
Loaded 8 documents.
Created 8 chunks.
Creating embeddings...
Created 8 embeddings.
ChromaDB collection 'zepto_policies' contains 8 records.

The exact output may vary depending on the installed library versions and ChromaDB state.

Limitations

This implementation intentionally keeps the architecture simple.

Each document is represented by one chunk.
The mock intent classifier relies on keyword matching.
The mock answer is a deterministic snippet from the closest retrieved document.
Retrieval always returns up to three results, without an explicit similarity threshold.
The optional real LLM depends on the availability and behavior of the configured Groq model.
The application currently exposes a single primary question-answer endpoint.

For a production implementation, chunk-level splitting, similarity thresholds, richer intent classification, authentication, logging, monitoring, rate limiting, and comprehensive automated tests could be added.

Future Improvements

Potential enhancements include:

Semantic intent classification instead of keyword matching.
Configurable chunk sizes and overlap.
Retrieval similarity thresholds.
Better source citations in answers.
Conversation/session memory.
Authentication and authorization.
Rate limiting.
Automated unit and integration tests.
Structured application logging.
Docker deployment.
Production database configuration.
Evaluation datasets for RAG accuracy.
LLM observability and tracing.
Improved fallback behavior when no relevant policy is found.
License

Add the appropriate project/company license here.

Summary

The Zepto Support Assistant follows a straightforward RAG architecture:

Zepto Policy Documents
↓
Sentence Transformer
↓
Embeddings
↓
ChromaDB
↓
User Question
↓
LangGraph Intent Classification
↓
┌───────────────┴───────────────┐
│ │
Policy Question General Question
│ │
▼ ▼
Retrieve Top 3 Direct Answer
│
▼
Mock / Real LLM
│
▼
Pydantic Validation
│
▼
FastAPI JSON Response

The application is therefore suitable as a compact demonstration of RAG + vector search + LangGraph routing + structured LLM output + FastAPI serving, while retaining a deterministic default mode for development and evaluation.
