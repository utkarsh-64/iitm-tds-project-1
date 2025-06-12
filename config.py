import os

USE_FAISS = os.getenv("USE_FAISS", "true").lower() == "true"
