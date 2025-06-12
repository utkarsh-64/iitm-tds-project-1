import os
import json
import glob
import markdown
import tiktoken
import numpy as np
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
import google.generativeai as genai

# Load Gemini API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Tokenizer for chunking
encoding = tiktoken.get_encoding("cl100k_base")

# Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Helper functions
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))
    return chunks

def get_gemini_embedding(text):
    res = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="RETRIEVAL_DOCUMENT"
    )
    return np.array(res["embedding"], dtype=np.float32)

###################################
# Load Markdown Files
###################################
markdown_data = []
md_files = glob.glob("markdown_files/*.md")

for md_file in md_files:
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
        url = "unknown"
        if md_content.startswith("---"):
            try:
                header, body = md_content.split("---", 2)[1:]
                meta_lines = header.strip().split("\n")
                for line in meta_lines:
                    if line.startswith("original_url:"):
                        url = line.split(":", 1)[1].strip()
                text = body
            except:
                text = md_content
        else:
            text = md_content

        chunks = chunk_text(text)
        for chunk in chunks:
            markdown_data.append({
                "source": "markdown",
                "url": url,
                "content": chunk
            })

###################################
# Load Discourse JSON
###################################
with open("discourse_posts.json", "r", encoding="utf-8") as f:
    discourse_posts = json.load(f)

topic_groups = {}
for post in discourse_posts:
    topic_id = post["topic_id"]
    topic_title = post.get("topic_title", "")
    topic_groups.setdefault(topic_id, {
        "topic_title": topic_title,
        "posts": []
    })
    topic_groups[topic_id]["posts"].append(post)

discourse_data = []
for topic_id, data in topic_groups.items():
    full_text = f"Thread Title: {data['topic_title']}\n\n"
    full_text += "\n\n---\n\n".join(post["content"] for post in data["posts"])
    chunks = chunk_text(full_text)
    for chunk in chunks:
        discourse_data.append({
            "source": "discourse",
            "url": data["posts"][0]["url"],
            "content": chunk
        })

###################################
# Combine Both Datasets
###################################
all_data = markdown_data + discourse_data

###################################
# Generate Embeddings with Gemini
###################################
print("Generating Gemini embeddings...")
embeddings = []
metadata = []

for item in tqdm(all_data):
    emb = get_gemini_embedding(item["content"])
    embeddings.append(emb)
    metadata.append(item)

embeddings_np = np.vstack(embeddings).astype("float32")

# Save everything
np.save("embeddings.npy", embeddings_np)
with open("metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)
faiss.write_index(index, "faiss.index")

print("✅ All embeddings generated & stored using Gemini!")