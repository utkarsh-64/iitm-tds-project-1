import os
import numpy as np
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load env variables
load_dotenv()

# Pinecone configs from env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # e.g. gcp-starter
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Gemini embedding dimensions
EMBEDDING_DIM = 768

class VectorDB:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        # Only run create if index does not exist
        if PINECONE_INDEX not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=PINECONE_INDEX,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(cloud="gcp", region=PINECONE_ENV)
            )

        self.index = self.pc.Index(PINECONE_INDEX)

    def upsert_embeddings(self, ids, embeddings, metadata_list):
        vectors = [
            {
                "id": str(id_),
                "values": emb.tolist(),
                "metadata": meta
            }
            for id_, emb, meta in zip(ids, embeddings, metadata_list)
        ]
        self.index.upsert(vectors=vectors)

    def query(self, query_embedding, k=5):
        query_embedding = query_embedding.tolist()
        response = self.index.query(
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
