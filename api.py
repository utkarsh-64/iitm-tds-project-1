import os
import json
import numpy as np
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import tiktoken
from pinecone import Pinecone
import openai
import concurrent.futures

# Load env vars
load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# ✅ v1.x OpenAI Client Config
client = openai.OpenAI(
    api_key=AIPROXY_TOKEN,
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
)

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Tokenizer (optional)
encoding = tiktoken.get_encoding("cl100k_base")

# FastAPI init
app = FastAPI()

# CORS setup (important for evaluator tests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

# Embed query (AIPROXY + padding)
def embed_query(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding

    # Pad to 2048 dims
    if len(embedding) < 2048:
        embedding = embedding + [0.0] * (2048 - len(embedding))

    return embedding

# Pinecone retrieval
def retrieve_top_k(query_embedding, k=3):
    response = index.query(
        vector=query_embedding,
        top_k=k,
        include_metadata=True
    )
    results = []
    for match in response["matches"]:
        meta = match["metadata"]
        results.append({
            "url": meta["url"],
            "content": meta["content"]
        })
    return results

# Generate answer
def generate_answer(question, retrieved_chunks):
    context = "\n\n".join([chunk['content'] for chunk in retrieved_chunks])
    prompt = f"""
You are a virtual teaching assistant.

Use the following context to answer the student's question as accurately as possible:

{context}

Question: {question}
Answer:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Generate snippet
def generate_snippet(question, chunk_text):
    snippet_prompt = f"""
You are a helpful assistant. Given the student's question: '{question}', summarize in one sentence why the following content is relevant to answer it:

Content: {chunk_text}

Summary:
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful summarizer."},
            {"role": "user", "content": snippet_prompt}
        ]
    )
    return response.choices[0].message.content

# Parallel snippet extraction
def extract_links_parallel(retrieved_chunks, question):
    links = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(generate_snippet, question, chunk["content"])
            for chunk in retrieved_chunks
        ]
        snippets = [future.result() for future in futures]

    for chunk, snippet in zip(retrieved_chunks, snippets):
        links.append({"url": chunk["url"], "text": snippet})
    
    return links

# ✅ Final /api route (Evaluator-safe)
@app.post("/api")
@app.post("/api/")
async def rag_api(req: QueryRequest):
    try:
        query_emb = embed_query(req.question)
        top_chunks = retrieve_top_k(query_emb, k=3)
        answer = generate_answer(req.question, top_chunks)
        links = extract_links_parallel(top_chunks, req.question)
        return {"answer": answer, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check (optional for debugging)
@app.get("/api")
@app.get("/api/")
async def health_check():
    return {"status": "API running."}
