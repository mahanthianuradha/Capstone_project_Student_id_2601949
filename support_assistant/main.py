# ============================================================
# MODULE 3 — ZEpto Support Assistant
# ============================================================
#
# This program:
#
# 1. Loads the eight Zepto policy documents.
# 2. Creates embeddings using Sentence Transformers.
# 3. Stores those embeddings in ChromaDB.
# 4. Uses LangGraph to route user questions.
# 5. Retrieves relevant policy documents when required.
# 6. Uses deterministic MOCK_LLM logic by default.
# 7. Optionally supports a real LLM when MOCK_LLM=0.
# 8. Validates final answers with Pydantic.
# 9. Provides a FastAPI POST /ask endpoint.
#
# ============================================================


# ------------------------------------------------------------
# Standard Python imports
# ------------------------------------------------------------

# os lets us read environment variables such as MOCK_LLM.
import os

# json lets us convert JSON strings into Python dictionaries.
import json

# re is used for a small amount of text processing.
import re

# Path provides safe cross-platform file and folder paths.
from pathlib import Path

# TypedDict describes the state used by LangGraph.
from typing import TypedDict, Literal


# ------------------------------------------------------------
# Third-party imports
# ------------------------------------------------------------

# FastAPI creates our web API.
from fastapi import FastAPI

# BaseModel and Field are used for request and response validation.
from pydantic import BaseModel, Field, ValidationError

# ChromaDB stores and searches vector embeddings.
import chromadb

# SentenceTransformer creates local text embeddings.
from sentence_transformers import SentenceTransformer

# LangGraph provides StateGraph and graph control.
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Find the directory containing this Python file.
# If main.py is:
# support_assistant/main.py
# BASE_DIR becomes:
# support_assistant/
BASE_DIR = Path(__file__).resolve().parent


# The document directory.
DOCS_DIR = BASE_DIR / "docs"


# ChromaDB will store its local database here.
CHROMA_DIR = BASE_DIR / "chroma_db"


# Name of the ChromaDB collection.
COLLECTION_NAME = "zepto_policies"


# The required local embedding model.
#
# This model creates numerical vectors representing the meaning
# of the document text.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# Randomness is not needed for the deterministic mock.
#
# The important environment variable is MOCK_LLM.
#
# If MOCK_LLM is missing:
#
#     mock mode
#
# If MOCK_LLM=1:
#
#     mock mode
#
# Only MOCK_LLM=0 enables the optional real LLM.
MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"


# Optional Groq model.

# This is used only when MOCK_LLM=0.
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.1-8b-instant",
)


# ============================================================
# 2. FASTAPI APPLICATION
# ============================================================

# Create the FastAPI application.
app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support assistant",
    version="1.0.0",
)


# ============================================================
# 3. PYDANTIC MODELS
# ============================================================

class AskRequest(BaseModel):
    """
    Request received by POST /ask.

    The client must send:

    {
        "query": "What are Zepto delivery charges?"
    }
    """

    # The user's question.
    query: str


class AnswerResponse(BaseModel):
    """
    Final structured response.

    Every answer must contain exactly these important fields:
    answer
    sources
    confidence
    """

    # The natural-language answer.
    answer: str

    # IDs of the documents/chunks used for the answer.
    #
    # This is empty for general questions.
    sources: list[str]

    # Confidence must always be between 0 and 1.
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


# ============================================================
# 4. LANGGRAPH STATE
# ============================================================

class AssistantState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Each node can read values from this dictionary and return
    updates to it.
    """

    # Original user question.
    query: str

    # Either policy_question or general_question.
    intent: str

    # Retrieved document/chunk information.
    retrieved_chunks: list[dict]

    # Final generated answer.
    answer: str

    # Source IDs.
    sources: list[str]

    # Final confidence score.
    confidence: float

    # Optional error information.
    error: str


# ============================================================
# 5. DOCUMENT INGESTION
# ============================================================

def load_documents() -> list[dict]:
    """
    Load all eight text documents.

    Returns a list containing:

        {
            "id": "doc_01",
            "text": "...",
            "source": "doc_01.txt"
        }
    """

    documents = []

    # Look for doc_01.txt through doc_08.txt.
    for number in range(1, 9):

        # Create the filename.
        file_path = DOCS_DIR / f"doc_{number:02d}.txt"

        # Make sure the required document exists.
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required corpus file not found: {file_path}"
            )

        # Read the complete text file.
        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        # Store the document information.
        documents.append(
            {
                "id": f"doc_{number:02d}",
                "text": text,
                "source": file_path.name,
            }
        )

    return documents


# ============================================================
# 6. CHUNKING
# ============================================================

def chunk_documents(
    documents: list[dict],
) -> list[dict]:
    """
    Split documents into chunks.

    The supplied documents are short, so one chunk per document
    is sufficient for this assignment.

    This still counts as a real chunking stage because every
    document becomes a searchable chunk with its own ID.
    """

    chunks = []

    # Process every document.
    for document in documents:

        # Create one chunk from the complete document.
        chunks.append(
            {
                # The chunk ID is the same as the document ID.
                "id": document["id"],

                # Store the text that will be embedded.
                "text": document["text"],

                # Store source metadata.
                "source": document["source"],
            }
        )

    return chunks


# ============================================================
# 7. LOAD EMBEDDING MODEL
# ============================================================

print("Loading local embedding model...")

# Load the open-source Sentence Transformer model.
#
# This does not require an LLM API key.
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)

print("Embedding model loaded.")


# ============================================================
# 8. CREATE CHROMADB CLIENT
# ============================================================

# PersistentClient stores ChromaDB data on disk.
#
# Therefore the embeddings remain available when the application
# restarts.
chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


# ============================================================
# 9. CREATE / LOAD CHROMA COLLECTION
# ============================================================

# Create the collection if it does not exist.
#
# "cosine" means cosine distance is used when comparing
# embedding vectors.
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    configuration={
        "hnsw": {
            "space": "cosine"
        }
    },
)


# ============================================================
# 10. BUILD THE VECTOR INDEX
# ============================================================

def build_vector_index() -> None:
    """
    Load documents, chunk them, embed them and store them
    in ChromaDB.

    This function can safely be run whenever the application
    starts because upsert updates existing records instead
    of creating duplicates.
    """

    print("Loading Zepto documents...")

    documents = load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    # Convert documents into searchable chunks.
    chunks = chunk_documents(documents)

    print(
        f"Created {len(chunks)} chunks."
    )

    # Extract the text from each chunk.
    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Creating embeddings...")

    # Convert each document into a numerical vector.
    #
    # normalize_embeddings=True makes the vectors suitable
    # for cosine similarity.
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    print(
        f"Created {len(embeddings)} embeddings."
    )

    # Extract the IDs.
    ids = [
        chunk["id"]
        for chunk in chunks
    ]

    # Store source information as metadata.
    metadatas = [
        {
            "source": chunk["source"]
        }
        for chunk in chunks
    ]

    # Insert or update the records in ChromaDB.
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )

    print(
        f"ChromaDB collection '{COLLECTION_NAME}' "
        f"contains {collection.count()} records."
    )


# Build the index when this application starts.
build_vector_index()


# ============================================================
# 11. STRUCTURED PROMPT TEMPLATE
# ============================================================

PROMPT_TEMPLATE = """
ROLE:
You are Zepto's policy support assistant.

CONTEXT:
Use ONLY the Zepto policy information provided in the
retrieved context below.

TASK:
Answer the user's question using the provided policy context.
If the context does not contain enough information, say that
the available policy context does not provide the answer.

FORMAT:
Return ONLY valid JSON using this structure:

{
  "answer": "string",
  "sources": ["doc_01"],
  "confidence": 0.0
}

The confidence value must be between 0 and 1.

LENGTH:
Keep the answer concise and normally within 2 to 4 sentences.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the
provided context. Do not invent Zepto policies, prices,
deadlines, fees or support procedures.

FEW-SHOT EXAMPLE:

Example user question:
How much is standard delivery below INR 149?

Example context:
Standard delivery is free on orders over INR 149; orders
below this threshold incur a flat INR 25 delivery fee.

Example valid answer:
{
  "answer": "Orders below INR 149 incur a flat INR 25 standard delivery fee.",
  "sources": ["doc_01"],
  "confidence": 1.0
}

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}
"""


# ============================================================
# 12. OPTIONAL REAL LLM
# ============================================================

def get_real_llm():
    """
    Create the optional real LLM.

    This function is NEVER called in the required mock mode.

    When MOCK_LLM=0, the user must provide:

        GROQ_API_KEY

    as an environment variable.
    """

    # Import only when actually needed.
    from langchain_groq import ChatGroq

    # Read the API key from the environment.
    api_key = os.getenv("GROQ_API_KEY")

    # Give a clear error if the user selected real mode
    # without supplying an API key.
    if not api_key:
        raise RuntimeError(
            "MOCK_LLM=0 requires the GROQ_API_KEY environment variable."
        )

    # Return the configured Groq chat model.
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=api_key,
    )


# ============================================================
# 13. PROMPT CREATION
# ============================================================

def create_prompt(
    query: str,
    context: str,
) -> str:
    """
    Insert the user's question and retrieved context into
    the structured prompt template.
    """

    return PROMPT_TEMPLATE.format(
        query=query,
        context=context,
    )


# ============================================================
# 14. PARSE REAL LLM JSON
# ============================================================

def parse_llm_response(
    raw_text: str,
) -> AnswerResponse:
    """
    Convert raw LLM output into our Pydantic response model.

    The model is expected to return JSON.

    Markdown code fences are removed if the model happens
    to return them.
    """

    # Remove unnecessary whitespace.
    cleaned = raw_text.strip()

    # Remove ```json and ``` if the model included code fences.
    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    # Convert the JSON text into a Python dictionary.
    data = json.loads(cleaned)

    # Pydantic validates:
    #
    # answer       -> string
    # sources      -> list of strings
    # confidence   -> number from 0 to 1
    return AnswerResponse.model_validate(data)


# ============================================================
# 15. REAL LLM WITH VALIDATION RETRIES
# ============================================================

def call_real_llm_with_validation(
    prompt: str,
) -> AnswerResponse:
    """
    Call the optional real LLM.

    If the response does not satisfy the Pydantic schema,
    retry up to two additional times.

    Therefore there can be a maximum of three attempts.
    """

    # Create the model.
    llm = get_real_llm()

    # Start with the original prompt.
    current_prompt = prompt

    # Keep the last error so it can be reported if all attempts fail.
    last_error = ""

    # Maximum three attempts:
    #
    # attempt 1 = original prompt
    # attempt 2 = corrective prompt
    # attempt 3 = corrective prompt again
    for attempt in range(3):

        try:
            # Send the prompt to the real LLM.
            response = llm.invoke(
                current_prompt
            )

            # Different LangChain versions normally expose
            # the response text through .content.
            raw_text = response.content

            # Try to validate the answer.
            return parse_llm_response(
                raw_text
            )

        except (
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as error:

            # Save the error for possible final reporting.
            last_error = str(error)

            # If this was the final attempt, stop retrying.
            if attempt == 2:
                break

            # Tell the LLM exactly what was wrong.
            current_prompt = f"""
Your previous response was invalid.

Validation error:
{last_error}

Return ONLY valid JSON with exactly these fields:

{{
  "answer": "string",
  "sources": ["string"],
  "confidence": 0.0
}}

Rules:
- answer must be a string.
- sources must be a list of strings.
- confidence must be between 0 and 1.
- Use only information from the supplied context.
- Do not add Markdown code fences.

Original request:

{prompt}
"""

    # If all three attempts fail, return a clearly marked
    # error response.
    return AnswerResponse(
        answer=(
            "ERROR: The real LLM response could not be "
            "validated after three attempts."
        ),
        sources=[],
        confidence=0.0,
    )


# ============================================================
# 16. INTENT CLASSIFICATION
# ============================================================

def classify_intent(
    state: AssistantState,
) -> dict:
    """
    LangGraph node 1.

    Decide whether the question is:

        policy_question

    or:

        general_question

    The required mock mode uses the exact keyword heuristic
    specified in the assignment.
    """

    # Read the user's question.
    query = state["query"]

    # Convert it to lowercase.
    lower_query = query.lower()

    # Exact required keywords from the assignment.
    policy_keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours",
    ]

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        # Check whether at least one policy keyword exists.
        is_policy_question = any(
            keyword in lower_query
            for keyword in policy_keywords
        )

        # Choose the correct intent.
        if is_policy_question:
            intent = "policy_question"
        else:
            intent = "general_question"

        print(
            f"[MOCK] classify_intent -> {intent}"
        )

        return {
            "intent": intent
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    # This branch is only reached when MOCK_LLM=0.
    llm = get_real_llm()

    classification_prompt = f"""
Classify the user's question into exactly one category:

policy_question
general_question

Return ONLY the category name.

Use policy_question when the question asks about Zepto's
delivery, returns, refunds, membership, tracking,
cancellation, gift cards, or customer support policies.

Otherwise return general_question.

User question:
{query}
"""

    response = llm.invoke(
        classification_prompt
    )

    # Normalize the model's response.
    result = response.content.strip().lower()

    # Make the routing robust.
    if "policy_question" in result:
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        "intent": intent
    }


# ============================================================
# 17. RETRIEVAL FUNCTION
# ============================================================

def retrieve_chunks(
    query: str,
) -> list[dict]:
    """
    Embed the query and retrieve the top three chunks from
    ChromaDB.

    This function ALWAYS performs real retrieval.

    MOCK_LLM does NOT disable embeddings or ChromaDB.
    """

    # Convert the user's question into an embedding.
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    # Search the ChromaDB collection.
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    # Chroma returns nested lists because the query API supports
    # multiple queries at once.
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    # Combine the returned information into easier-to-use
    # dictionaries.
    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):

        retrieved.append(
            {
                "id": chunk_id,
                "text": document,
                "source": metadata["source"],
                "distance": float(distance),
            }
        )

    return retrieved


# ============================================================
# 18. RETRIEVE AND ANSWER
# ============================================================

def retrieve_and_answer(
    state: AssistantState,
) -> dict:
    """
    LangGraph node 2.

    This node:
    
    1. Retrieves the top three policy chunks.
    2. Generates an answer.
    
    Retrieval happens in BOTH modes.
    
    Only answer generation changes with MOCK_LLM.
    """

    query = state["query"]

    # --------------------------------------------------------
    # REAL RETRIEVAL
    # --------------------------------------------------------

    retrieved_chunks = retrieve_chunks(
        query
    )

    # Safety check.
    if not retrieved_chunks:

        return {
            "retrieved_chunks": [],
            "answer": (
                "No relevant policy information was found."
            ),
            "sources": [],
            "confidence": 0.0,
        }

    # Source IDs of all retrieved chunks.
    source_ids = [
        chunk["id"]
        for chunk in retrieved_chunks
    ]

    # Combine retrieved documents into one context string.
    context = "\n\n".join(
        [
            f"[{chunk['id']}] {chunk['text']}"
            for chunk in retrieved_chunks
        ]
    )

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        # The assignment specifically requests a canned
        # response based on the most similar chunk.
        #
        # The first result is the closest result returned by
        # ChromaDB.
        top_chunk_snippet = (
            retrieved_chunks[0]["text"][:200]
        )

        answer = (
            "Based on the retrieved context: "
            + top_chunk_snippet
        )

        print(
            "[MOCK] retrieve_and_answer -> "
            "retrieval + canned answer"
        )

        return {
            "retrieved_chunks": retrieved_chunks,
            "answer": answer,
            "sources": source_ids,
            "confidence": 1.0,
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    # Create the structured prompt containing:
    #
    # Role
    # Context
    # Task
    # Format
    # Length
    # Negative constraint
    # Few-shot example
    prompt = create_prompt(
        query,
        context,
    )

    # Ask the real LLM and validate its output.
    validated_response = (
        call_real_llm_with_validation(
            prompt
        )
    )

    # Return the validated information to the graph state.
    return {
        "retrieved_chunks": retrieved_chunks,
        "answer": validated_response.answer,
        "sources": validated_response.sources,
        "confidence": validated_response.confidence,
    }


# ============================================================
# 19. DIRECT ANSWER
# ============================================================

def direct_answer(
    state: AssistantState,
) -> dict:
    """
    LangGraph node 3.

    This node handles general questions that do not require
    retrieval from the policy corpus.
    """

    query = state["query"]

    # --------------------------------------------------------
    # REQUIRED MOCK MODE
    # --------------------------------------------------------

    if MOCK_LLM:

        # Required fixed canned response.
        answer = (
            "I can only answer questions about Zepto policies "
            "right now."
        )

        print(
            "[MOCK] direct_answer -> canned answer"
        )

        return {
            "answer": answer,
            "sources": [],
            "confidence": 1.0,
        }

    # --------------------------------------------------------
    # OPTIONAL REAL LLM MODE
    # --------------------------------------------------------

    # Direct questions do not use retrieval.
    prompt = f"""
ROLE:
You are Zepto's support assistant.

CONTEXT:
There is no retrieved policy context for this question.

TASK:
Answer the user's question helpfully.

FORMAT:
Return ONLY valid JSON:

{{
  "answer": "string",
  "sources": [],
  "confidence": 0.0
}}

LENGTH:
Keep the answer concise.

NEGATIVE CONSTRAINT:
Do not invent specific Zepto policies or facts.

USER QUESTION:
{query}
"""

    validated_response = (
        call_real_llm_with_validation(
            prompt
        )
    )

    return {
        "answer": validated_response.answer,
        "sources": [],
        "confidence": validated_response.confidence,
    }


# ============================================================
# 20. CONDITIONAL ROUTER
# ============================================================

def route_after_classification(
    state: AssistantState,
) -> Literal[
    "retrieve_and_answer",
    "direct_answer",
]:
    """
    Decide which LangGraph node should execute next.

    IMPORTANT:
    This routing logic itself does not depend on MOCK_LLM.

    The classification node decides the intent.
    This function simply follows that decision.
    """

    if state["intent"] == "policy_question":

        return "retrieve_and_answer"

    return "direct_answer"


# ============================================================
# 21. BUILD LANGGRAPH
# ============================================================

# Create a graph builder using our TypedDict state.
graph_builder = StateGraph(
    AssistantState
)


# Add the three required named nodes.
graph_builder.add_node(
    "classify_intent",
    classify_intent,
)

graph_builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer,
)

graph_builder.add_node(
    "direct_answer",
    direct_answer,
)


# ------------------------------------------------------------
# Graph entry point
# ------------------------------------------------------------

# Every request starts with classify_intent.
graph_builder.add_edge(
    START,
    "classify_intent",
)


# ------------------------------------------------------------
# Conditional edge
# ------------------------------------------------------------

# After classification, route the request to one of two nodes.
graph_builder.add_conditional_edges(
    "classify_intent",
    route_after_classification,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer",
    },
)


# ------------------------------------------------------------
# End points
# ------------------------------------------------------------

# Both answer nodes finish the graph.
graph_builder.add_edge(
    "retrieve_and_answer",
    END,
)

graph_builder.add_edge(
    "direct_answer",
    END,
)


# Compile the graph so it can be invoked.
graph = graph_builder.compile()


# ============================================================
# 22. FASTAPI ENDPOINT
# ============================================================

@app.post(
    "/ask",
    response_model=AnswerResponse,
)
def ask(
    request: AskRequest,
) -> AnswerResponse:
    """
    POST /ask

    Example request:

    {
        "query": "How much is delivery below INR 149?"
    }
    """

    # Remove accidental whitespace.
    query = request.query.strip()

    # Do not allow an empty question.
    if not query:
        return AnswerResponse(
            answer="Please provide a question.",
            sources=[],
            confidence=0.0,
        )

    # Create the initial LangGraph state.
    initial_state: AssistantState = {
        "query": query
    }

    # Run the graph.
    final_state = graph.invoke(
        initial_state
    )

    # Construct the final Pydantic response.
    #
    # This provides one final validation layer before
    # FastAPI returns JSON to the client.
    response = AnswerResponse(
        answer=final_state.get(
            "answer",
            "No answer was generated.",
        ),
        sources=final_state.get(
            "sources",
            [],
        ),
        confidence=final_state.get(
            "confidence",
            0.0,
        ),
    )

    return response


# ============================================================
# 23. HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    """
    Simple health-check endpoint.

    It is useful to confirm that the API is running.
    """

    return {
        "message": "Zepto Support Assistant is running.",
        "mock_llm": MOCK_LLM,
        "collection": COLLECTION_NAME,
    }


# ============================================================
# END OF FILE
# ============================================================