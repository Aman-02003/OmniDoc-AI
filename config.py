import os

API_KEY = os.getenv("GOOGLE_API_KEY", "")
NEW_API_KEY = API_KEY
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
# ─────────────────────────────────────────────
# LLM MODEL — Choose based on your RAM:
# ─────────────────────────────────────────────
# Your RAM:  6GB  → use qwen2:1.5b  (~1.1 GB)
# Your RAM:  8GB  → use qwen2:7b    (~5.6 GB)
# Your RAM: 16GB  → use mistral:7b  (~4.1 GB)
# ─────────────────────────────────────────────
# LLM_MODEL = "qwen2:1.5b"   # ← CHANGED from qwen2:7b (too large for your RAM)

# Embedding Model — lightweight, stays the same
LOCAL_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Other settings
Embedding_MODEL = "models/gemini-embedding-001"
MODEL = "models/gemini-2.5-flash"
# MODEL = "models/gemini-1.5-flash"
PATH = "/home/stpl/Desktop/AI Agent/Data/company_files"
MODE = "condense_plus_context"