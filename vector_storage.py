import os
import json
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env
load_dotenv()

# Load API keys from env
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

# Configure OpenAI for AIPROXY
openai.api_key = AIPROXY_TOKEN
openai.base_url = "https://aiproxy.sanand.workers.dev/openai/v1"

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Load metadata.json file (contains your scraped data)
with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Helper function to embed and pad
def get_padded_embedding(text):
    client = openai.OpenAI(
    api_key=AIPROXY_TOKEN,
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"
    )

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    embedding = response.data[0].embedding

    # Pad to 2048 dimensions
    if len(embedding) < 2048:
        embedding = embedding + [0.0] * (2048 - len(embedding))
    
    return embedding

# Loop through data and upload to Pinecone
for i, item in enumerate(tqdm(metadata, desc="Uploading to Pinecone")):
    text = item['content']
    url = item['url']

    # Generate embedding
    embedding = get_padded_embedding(text)

    # Upload to Pinecone
    index.upsert(
        vectors=[(str(i), embedding, {"content": text, "url": url})]
    )