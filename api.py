import os
import json
import numpy as np
from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import tiktoken
from pinecone import Pinecone
import concurrent.futures

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

genai.configure(api_key=GEMINI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Tokenizer (not strictly used but keeping)
encoding = tiktoken.get_encoding("cl100k_base")

app = FastAPI()

# Enable CORS (important for evaluator testing)
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

# Gemini Embedding function
def embed_query(text):
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_QUERY"
    )
    return np.array(res["embedding"], dtype=np.float32).tolist()

# Pinecone search function
def retrieve_top_k(query_embedding, k=5):
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

# Gemini Answer function
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

# Snippet generation function (not async but parallelizable)
def generate_snippet(question, chunk_text):
    snippet_prompt = f"""
You are a helpful assistant. Given the student's question: '{question}', summarize in one sentence why the following content is relevant to answer it:

Content: {chunk_text}

Summary:
"""
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(snippet_prompt)
    return response.text.strip()

# Fully parallel snippet generation
def extract_links_parallel(retrieved_chunks, question):
    links = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(generate_snippet, question, chunk["content"])
            for chunk in retrieved_chunks
        ]
        snippets = [future.result() for future in futures]

    for chunk, snippet in zip(retrieved_chunks, snippets):
        links.append({"url": chunk["url"], "text": snippet})
    
    return links

# ✅ POST endpoint as required
@app.post("/api")
@app.post("/api/")
async def rag_api(req: QueryRequest):
    try:
        query_emb = embed_query(req.question)
        top_chunks = retrieve_top_k(query_emb, k=5)
        answer = generate_answer(req.question, top_chunks)
        links = extract_links_parallel(top_chunks, req.question)
        return {"answer": answer, "links": links}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional health check
@app.get("/api")
@app.get("/api/")
async def health_check():
    return {"status": "API running. Please use POST request."}
