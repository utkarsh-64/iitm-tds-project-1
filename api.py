import os
import json
import base64
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import tiktoken

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Load FAISS index and metadata
embeddings = np.load("embeddings.npy")
metadata = json.load(open("metadata.json", "r", encoding="utf-8"))
index = faiss.read_index("faiss.index")

# Tokenizer for chunking prompt if needed
encoding = tiktoken.get_encoding("cl100k_base")

# FastAPI initialization
app = FastAPI()

# Enable CORS for all origins (can be restricted in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # image in base64 (not used right now)

# Helper: Embed query using Gemini

def embed_query(text):
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_QUERY"
    )
    return np.array(res["embedding"], dtype=np.float32)

# Helper: Similarity Search

def retrieve_top_k(query_embedding, k=5):
    query_embedding = query_embedding.reshape(1, -1).astype("float32")
    D, I = index.search(query_embedding, k)
    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

# Helper: Generate answer with Gemini Flash

def generate_answer(question, retrieved_chunks):
    context = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])
    prompt = f"""
You are a virtual teaching assistant.

Use the following context to answer the student's question as accurately as possible:

{context}

Question: {question}
Answer:
"""

    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)
    return response.text.strip()

# Helper: Generate snippet for each link using Gemini Flash

def generate_snippet(question, chunk_text):
    snippet_prompt = f"""
You are a helpful assistant. Given the student's question: '{question}', summarize in one sentence why the following content is relevant to answer it:

Content: {chunk_text}

Summary:
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(snippet_prompt)
    return response.text.strip()

# Helper: Extract links with smarter snippets

def extract_links(retrieved_chunks, question):
    links = []
    for chunk in retrieved_chunks:
        snippet = generate_snippet(question, chunk["content"])
        links.append({
            "url": chunk["url"],
            "text": snippet
        })
    return links

# API Endpoint for both /api and /api/
@app.post("/api")
@app.post("/api/")
async def rag_api(req: QueryRequest):
    try:
        query_emb = embed_query(req.question)
        top_chunks = retrieve_top_k(query_emb, k=5)
        answer = generate_answer(req.question, top_chunks)
        links = extract_links(top_chunks, req.question)

        response = {
            "answer": answer,
            "links": links
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
