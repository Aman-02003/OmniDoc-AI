"""
Run this script from your project folder:
    python3 debug_check.py

This will tell you EXACTLY what is broken.
"""

import chromadb

print("=" * 60)
print("STEP 1: CHECKING CHROMADB COLLECTIONS")
print("=" * 60)

try:
    db = chromadb.PersistentClient(path="./chroma_db")
    collections = db.list_collections()
    
    if not collections:
        print("❌ CRITICAL: No collections found in chroma_db!")
        print("   → Your documents were NEVER indexed.")
        print("   → Fix: Delete chroma_db/ and docstore.json, then re-run app.py")
    else:
        for col in collections:
            c = db.get_collection(col.name)
            count = c.count()
            print(f"Collection: '{col.name}' → {count} chunks")
            if count == 0:
                print(f"   ❌ EMPTY COLLECTION! Documents not indexed.")
            else:
                print(f"   ✅ Has data.")
                # Show a sample chunk
                sample = c.get(limit=1)
                print(f"   Sample chunk: {sample['documents'][0][:150]}...")
except Exception as e:
    print(f"❌ ChromaDB Error: {e}")

print()
print("=" * 60)
print("STEP 2: CHECKING RETRIEVAL SCORES")
print("=" * 60)

try:
    from llama_index.core import Settings, StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.chroma import ChromaVectorStore

    print("Loading embedding model (bge-base)...")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.embed_model = embed_model

    db = chromadb.PersistentClient(path="./chroma_db")
    
    # Try both collection names
    collection_names = ["lottery_terms_collection", "ibc_intelligent_collection"]
    working_collection = None
    
    for name in collection_names:
        try:
            col = db.get_collection(name)
            if col.count() > 0:
                working_collection = name
                print(f"✅ Found working collection: '{name}' with {col.count()} chunks")
                break
        except:
            pass
    
    if not working_collection:
        print("❌ No working collection with data found!")
        print("   SOLUTION: Delete chroma_db/ and docstore.json and re-run")
    else:
        chroma_col = db.get_collection(working_collection)
        vector_store = ChromaVectorStore(chroma_collection=chroma_col)
        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = index.as_retriever(similarity_top_k=5)
        
        test_queries = [
            "Lucky Day lottery",
            "forecast lottery",
            "prize winning amount",
            "eligibility age"
        ]
        
        print()
        for query in test_queries:
            print(f"Query: '{query}'")
            nodes = retriever.retrieve(query)
            if nodes:
                print(f"  Top score: {nodes[0].score:.4f}")
                print(f"  Top text:  {nodes[0].node.text[:100]}...")
                print(f"  → {'✅ Would PASS' if nodes[0].score >= 0.30 else '❌ Would FAIL'} threshold 0.30")
                print(f"  → {'✅ Would PASS' if nodes[0].score >= 0.60 else '❌ Would FAIL'} threshold 0.60")
            else:
                print("  ❌ NO NODES RETURNED AT ALL!")
            print()

except Exception as e:
    print(f"❌ Retrieval Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("STEP 3: CHECKING OLLAMA")
print("=" * 60)

import subprocess
result = subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], 
                       capture_output=True, text=True)
if result.returncode == 0 and result.stdout:
    print(f"✅ Ollama is running")
    print(f"   Response: {result.stdout[:200]}")
else:
    print("❌ Ollama is NOT running or not accessible")
    print("   Fix: Run 'ollama serve' in a separate terminal")

print()
print("=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)