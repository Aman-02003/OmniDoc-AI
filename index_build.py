import time
import os
import chromadb
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from config import API_KEY, Embedding_MODEL, LOCAL_EMBEDDING_MODEL
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def build_index(nodes):
    if not API_KEY:
        raise RuntimeError("Set GOOGLE_API_KEY in your environment before building the index.")

    # 1. Initialize the Gemini Embedding Model
    embed_model = GoogleGenAIEmbedding(
        model_name=Embedding_MODEL, 
        api_key=API_KEY,
        embed_batch_size=20
    )
    
    # 2. CRITICAL: Global Settings configuration
    # This prevents LlamaIndex from looking for an OpenAI API key
    Settings.embed_model = embed_model
    # Set chunk size to match your process.py logic just in case
    Settings.chunk_size = 512

    # 3. Setup ChromaDB Persistence
    db_path = "./chroma_db"
    db = chromadb.PersistentClient(path=db_path)
    
    # Optional: Clear existing collection if you want to ensure all 272 are re-indexed
    # Remove these 3 lines if you want to keep existing data
    if db.get_collection("ibc_manual_collection"):
        db.delete_collection(name="ibc_manual_collection")
        print("Existing collection deleted to ensure a clean re-index of all 272 chunks.")

    chroma_collection = db.get_or_create_collection(name="ibc_manual_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Check if data already exists
    if chroma_collection.count() > 0:
        print(f"Loading {chroma_collection.count()} vectors from ChromaDB...")
        return VectorStoreIndex.from_vector_store(
            vector_store, 
            storage_context=storage_context,
            embed_model=embed_model # Pass explicitly for safety
        )

    # 5. Manual Batch Indexing for all 272 nodes
    print(f"Starting manual batch indexing for {len(nodes)} nodes...")
    
    # Initialize an EMPTY index explicitly using our embed_model
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=embed_model)
    
    batch_size = 50 
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i : i + batch_size]
        print(f"Processing batch {i//batch_size + 1}... (Nodes {i} to {min(i + batch_size, len(nodes))})")
        
        # Insert the batch into the index
        index.insert_nodes(batch)
        
        # Cooldown to respect Google API rate limits
        if i + batch_size < len(nodes):
            print("Cooldown: Waiting 60 seconds to reset API quota...")
            time.sleep(60)
            
    print(f"Success! All {len(nodes)} chunks are now stored in ChromaDB.")
    return index