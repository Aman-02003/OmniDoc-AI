"""
Run this to see EXACT scores for your queries:
    python3 score_check.py
"""

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

print("Loading embedding model...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
Settings.embed_model = embed_model

db = chromadb.PersistentClient(path="./chroma_db")

# Find collection with data
working = None
for col in db.list_collections():
    c = db.get_collection(col.name)
    if c.count() > 0:
        working = col.name
        print(f"✅ Collection '{col.name}' has {c.count()} chunks\n")
        break

if not working:
    print("❌ No data found! Run: rm -rf chroma_db/ && python3 main.py")
    exit()

chroma_col = db.get_collection(working)
vector_store = ChromaVectorStore(chroma_collection=chroma_col)
index = VectorStoreIndex.from_vector_store(vector_store)
retriever = index.as_retriever(similarity_top_k=3)

queries = [
    "Lucky Day",
    "Forecast Lotteries",
    "FIXED PREMIUMS",
    "prize winning amount",
    "eligibility age bet",
    "Mega Sena draw",
]

print("=" * 65)
print(f"{'QUERY':<30} {'TOP SCORE':<12} {'PASS 0.25?'}")
print("=" * 65)

for q in queries:
    try:
        nodes = retriever.retrieve(q)
        if nodes:
            score = nodes[0].score
            status = "✅ YES" if score >= 0.25 else "❌ NO"
            print(f"{q:<30} {score:<12.4f} {status}")
            print(f"  → Chunk: {nodes[0].node.text[:80].strip()}...")
        else:
            print(f"{q:<30} {'NO NODES':<12} ❌ NO")
    except Exception as e:
        print(f"{q:<30} ERROR: {e}")
    print()