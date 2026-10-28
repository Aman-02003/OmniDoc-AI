import os
import json
import hashlib
import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Path to track which files have already been indexed
FILE_TRACKER = "./indexed_files.json"


def get_file_hash(filepath):
    """Generate MD5 hash of a file to detect changes."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_indexed_files():
    """Load the record of already-indexed files."""
    if os.path.exists(FILE_TRACKER):
        with open(FILE_TRACKER, "r") as f:
            return json.load(f)
    return {}


def save_indexed_files(indexed):
    """Save the updated record of indexed files."""
    with open(FILE_TRACKER, "w") as f:
        json.dump(indexed, f, indent=2)
    print(f"[INDEX.py] Saved file tracker: {FILE_TRACKER}")


def build_index(nodes):

    # 1. Embedding model
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.embed_model = embed_model

    # 2. ChromaDB setup
    db = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "IBC_Document"
    chroma_collection = db.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    current_count = chroma_collection.count()
    print(f"[INDEX.py] Current DB chunk count: {current_count}")

    # 3. Check which files need indexing using FILE HASH TRACKER
    indexed_files = load_indexed_files()
    nodes_to_index = []
    already_indexed = []

    for node in nodes:
        # Get the source file for this node
        file_name = node.metadata.get("file_name", "unknown")
        file_path = node.metadata.get("file_path", "")

        # Generate hash of the source file
        if file_path and os.path.exists(file_path):
            file_hash = get_file_hash(file_path)
        else:
            # Fallback: use filename only
            file_hash = hashlib.md5(file_name.encode()).hexdigest()

        # Check if this file version is already indexed
        if file_name in indexed_files and indexed_files[file_name] == file_hash:
            already_indexed.append(file_name)
        else:
            nodes_to_index.append((node, file_name, file_hash))

    # Report what we found
    unique_already = list(set(already_indexed))
    unique_to_index = list(set([x[1] for x in nodes_to_index]))

    if unique_already:
        print(f"[INDEX.py] ✅ Already indexed (skipping): {unique_already}")
    if unique_to_index:
        print(f"[INDEX.py] 🆕 New/changed files to index: {unique_to_index}")

    # 4. Only index NEW nodes
    if not nodes_to_index:
        print(f"[INDEX.py] ✅ No new files — loading existing index.")
        return VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=StorageContext.from_defaults(vector_store=vector_store)
        )

    new_nodes = [item[0] for item in nodes_to_index]
    print(f"[INDEX.py] Indexing {len(new_nodes)} new chunks...")

    # 5. Run pipeline — only embed, never re-chunk
    pipeline = IngestionPipeline(
        transformations=[embed_model],
        vector_store=vector_store,
    )

    pipeline.run(nodes=new_nodes, show_progress=True)

    # 6. Update file tracker — mark these files as indexed
    for _, file_name, file_hash in nodes_to_index:
        indexed_files[file_name] = file_hash

    save_indexed_files(indexed_files)

    final_count = chroma_collection.count()
    print(f"[INDEX.py] ✅ Done. DB now has {final_count} chunks.")

    return VectorStoreIndex.from_vector_store(
        vector_store,
        storage_context=StorageContext.from_defaults(vector_store=vector_store)
    )