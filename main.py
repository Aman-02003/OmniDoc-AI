from read import load_documents
from config import PATH
# from my_token import get_token
from process import process_documents
# from index_build import build_index
# from INDEXING import build_index
from INDEX import build_index
import chromadb
from Chat.chat_logic import get_chat_engine

db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="IBC_Document")

if collection.count() == 0:
    print("Empty DB. Performing initial indexing...")
    documents = load_documents(PATH)
    nodes = process_documents(documents)
    print(f"Success! Created {len(nodes)} smart chunks (Nodes).")
    unique_files = {doc.metadata['file_name'] for doc in documents}
    print(f"--- Files detected in {PATH} ---")
    for file in unique_files:
        print(f"Found: {file}")
    print(f"\nTotal Chunks (Nodes) created: {len(nodes)}")
    index = build_index(nodes)
else:
    # DB has Data. We can either:
    # 1. Load it instanly 
    # 2. process doc to see it there is a new file
    print("Checking for new files in Data Folder...")
    documents = load_documents(PATH)
    if not documents or len(documents) == 0:
        print("❌ CRITICAL: No documents were parsed. Check LlamaParse API quota or connection.")
        exit() # Don't proceed to 0 chunks
    nodes = process_documents(documents)
    print(f"Success! Created {len(nodes)} smart chunks (Nodes).")
    unique_files = {doc.metadata['file_name'] for doc in documents}
    print(f"--- Files detected in {PATH} ---")
    for file in unique_files:
        print(f"Found: {file}")
    print(f"\nTotal Chunks (Nodes) created: {len(nodes)}")
    index = build_index(nodes)

chat_bot = get_chat_engine(index)
print("\n" + "="*30)
# print("  IBC CHATBOT IS ONLINE")
# print("  Type 'exit', 'bye', 'quit' to quit")
# print("="*30)
# while True:
#     user_query = input("\nYou: ")
#     if user_query.lower() in ["exit", "quit", "bye"]:
#         print("Assistant: Goodbye! Have a great day.")
#         break
#     response = chat_bot(user_query)
#     print(f"\nAssistant: {response}")
# 
    # Use .chat() to maintain memory (don't use .query() here!)
    

print(index)

































# print(f"First Node Source: {nodes[0].metadata['file_name']} - Page {nodes[0].metadata['page_label']}")

"""
import json # To make the output look pretty

# ... (after you have created 'nodes')

# 1. Select the first node to inspect
sample_node = nodes[79]

# 2. Convert the node to a dictionary for visualization
node_dict = sample_node.dict()

# 3. Print it in a beautiful, structured format
print("\n--- VISUALIZING NODE FORMAT ---")
print(json.dumps({
    "Node ID": node_dict['id_'],
    "Text Snippet": node_dict['text'][:150] + "...", # Showing first 150 chars
    "Metadata": node_dict['metadata'],
    "Relationships": node_dict['relationships']
}, indent=4))
# print(documents)
"""
# tokens = get_token(nodes)