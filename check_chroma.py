import streamlit as st
import chromadb
import pandas as pd

st.set_page_config(layout="wide", page_title="ChromaDB Viewer")

st.title("🔍 IBC Manual Collection Browser")

# This path matches your index_building.py
DB_PATH = "./chroma_db" 
COLLECTION_NAME = "IBC_Document"

try:
    # Initialize Chroma Client
    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Get your specific collection
    collection = client.get_collection(name=COLLECTION_NAME)
    
    # Fetch all data
    data = collection.get()
    
    st.write(f"### Total Chunks Found: {len(data['ids'])}")

    # Format data for a clean table view
    df = pd.DataFrame({
        "ID": data["ids"],
        "Content": data["documents"],
        "Source File": [m.get('file_name', 'N/A') for m in data["metadatas"]],
        "Page": [m.get('page_label', 'N/A') for m in data["metadatas"]]
    })

    # Display the table
    st.dataframe(df, use_container_width=True)

    # Search bar logic
    search_query = st.text_input("Search inside your chunks:")
    if search_query:
        results = collection.query(query_texts=[search_query], n_results=5)
        st.write("### Top 5 Relevant Search Results:")
        st.write(results["documents"])

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Make sure you are running this from the folder that contains the 'chroma_db' folder.")