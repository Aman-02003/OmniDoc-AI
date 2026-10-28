from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

def process_documents(documents):
    """
    FIX: Reduced chunk_size from 1024 → 512 for legal/lottery documents.
    Dense legal clauses need smaller chunks for precise retrieval.
    FIX: Chunking only happens HERE — removed duplicate SentenceSplitter from INDEX.py pipeline.
    """
    splitter = SentenceSplitter(
        chunk_size=512,       # Smaller = more precise retrieval for legal text
        chunk_overlap=50,     # Reduced overlap to avoid redundancy
        include_metadata=True
    )
    Settings.text_splitter = splitter
    nodes = Settings.text_splitter.get_nodes_from_documents(documents)
    print(f"[process.py] Created {len(nodes)} chunks (chunk_size=512)")
    return nodes