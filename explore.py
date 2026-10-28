import streamlit as st
import chromadb
import pandas as pd
from config import LOCAL_EMBEDDING_MODEL

def show_db_explorer():
    st.title("📂 IBC Dataset Explorer")
    st.markdown("Use this tool to verify what is currently stored in your Vector DB.")

    # 1. Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # 2. Select Collection (Matches your INDEX.py)
    collection_name = "ibc_intelligent_collection"
    try:
        collection = client.get_collection(name=collection_name)
    except:
        st.error(f"Collection '{collection_name}' not found. Please index data first.")
        return

    # 3. Fetch Data
    # Fetching everything: documents, metadatas, and IDs
    results = collection.get(include=["documents", "metadatas"])
    
    if not results['ids']:
        st.warning("The database is empty.")
        return

    # 4. Process for Streamlit
    data = []
    for i in range(len(results['ids'])):
        row = {
            "Chunk ID": results['ids'][i],
            "Content Preview": results['documents'][i],
            "File Name": results['metadatas'][i].get('file_name', 'N/A'),
            "Page": results['metadatas'][i].get('page_label', 'N/A'),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # 5. UI Elements
    col1, col2 = st.columns(2)
    col1.metric("Total Chunks", len(df))
    col2.metric("Source Files", df['File Name'].nunique())

    # Search Filter
    search_query = st.text_input("🔍 Search within Database Chunks:", "")
    if search_query:
        df = df[df['Content Preview'].str.contains(search_query, case=False)]

    # Interactive Table
    st.dataframe(
        df, 
        use_container_width=True, 
        column_config={
            "Content Preview": st.column_config.TextColumn("Content Preview", width="large")
        }
    )

if __name__ == "__main__":
    show_db_explorer()