import asyncio
import nest_asyncio
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
nest_asyncio.apply()

import streamlit as st
from read import load_documents
from process import process_documents
from INDEX import build_index
from Chat.chat_logic import get_chat_engine
from config import PATH
import chromadb
import json
import os
import time


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Lottery T&C Assistant",
    layout="wide", # Wider layout for a more modern feel
    page_icon="🎰",
    initial_sidebar_state="expanded"
)

# --- PROFESSIONAL STYLING (CSS) ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #f9fbfd;
    }

    /* User Message Style */
    [data-testid="stChatMessageUser"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e6ed !important;
        border-radius: 15px !important;
        padding: 15px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }

    /* Assistant Message Style */
    [data-testid="stChatMessageAssistant"] {
        background-color: #f0f4f8 !important;
        border-left: 5px solid #007bff !important; /* Professional blue accent */
        border-radius: 0 15px 15px 0 !important;
        padding: 15px !important;
    }

    /* Hide Streamlit Header/Footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #eee;
    }

    /* Input box styling */
    .stChatInputContainer {
        padding-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_system():
    try:
        """Builds or loads the index once per session."""
        db = chromadb.PersistentClient(path="./chroma_db")
        collection = db.get_or_create_collection(name="IBC_Document")
        count = collection.count()
        print(f"[app.py] DB has {count} chunks on initialization.")
    except Exception as e:
        print(f"[app.py] ❌ Error connecting to DB: {type(e).__name__}: {e}")
        return None
    
    
    with st.spinner("✨ Initializing Intelligence Engine..."):
        documents = load_documents(PATH)
        if not documents:
            st.error("No documents found in the specified path. Please check your data folder.")
            return None
        nodes = process_documents(documents)
        index = build_index(nodes)
    
    engine = get_chat_engine(index)
    return engine

# --- SIDEBAR UI ---
with st.sidebar:
    st.image("/home/stpl/Desktop/AI Agent/Data/skilrock_logo.jpeg", width=65) # Placeholder Logo
    st.title("Skilrock IBC Assistant")
    st.markdown("---")
    
    # Status Metrics
    try:
        db = chromadb.PersistentClient(path="./chroma_db")
        col = db.get_collection("IBC_Document")
        count = col.count()
        st.success(f"Database: {count} Chunks loaded!")
        st.metric(label="Knowledge Base", value=f"{count} Chunks")
    except:
        st.error("Database connection failed.")
        count = 0

    # File Inventory
    with st.expander("📄 Managed Documents", expanded=False):
        if os.path.exists("./indexed_files.json"):
            with open("./indexed_files.json") as f:
                indexed = json.load(f)
            st.info(f"{len(indexed)} files indexed:")
            for fname in indexed:
                st.caption(f"• {fname}")
        else:
            st.warning("No files found.")

    st.markdown("---")
    
    # Action Buttons
    col1, col2 = st.columns(2)
    if col1.button("🔄 Reload", help="Refresh data"):
        st.cache_resource.clear()
        st.rerun()
    if col2.button("🗑️ Reset", help="Wipe Database", type="secondary"):
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        for f in ["./docstore.json", "./indexed_files.json"]:
            if os.path.exists(f):
                os.remove(f)
        st.cache_resource.clear()
        st.rerun()

# --- MAIN INTERFACE ---
chat_bot = initialize_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Welcome Header
st.title("Skilrock IBC Terms & Conditions Assistant")
st.info("I am an AI trained specifically on official Lottery Terms and Conditions. Ask me about eligibility, prize structures, or legal deadlines.")

# --- DISPLAY CHAT HISTORY ---
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- CHAT INPUT & RESPONSE ---
if prompt := st.chat_input("Ask a question about the Terms of Reference..."):
    if chat_bot is None:
        st.error("System not initialized. Please check your data folder and restart the app.")
    else:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            status_placeholder = st.empty()
            with status_placeholder.status("🔍 Analyzing documents...", expanded=True) as status:
                # Simulate a more professional "thinking" process
                st.write("Retrieving relevant clauses...")
                response = chat_bot(prompt)
                time.sleep(0.5) 
                st.write("Verifying compliance...")
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            
            # Use a slight typewriter effect for the response
            full_response = ""
            placeholder = st.empty()
            # If your chat_bot doesn't support streaming, we simulate it for UX:
            for word in response.split(" "):
                full_response += word + " "
                placeholder.markdown(full_response + "▌")
                time.sleep(0.02)
            placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})

# Footer suggestions
# if not st.session_state.messages:
    # st.markdown("### Suggested Topics")
    # cols = st.columns(3)
    # if cols[0].button("High Availability (HA)"):
        # st.info("Ask: 'What are the HA considerations mentioned?'")
    # if cols[1].button("Eligibility"):
        # st.info("Ask: 'Who is eligible to participate?'")
    # if cols[2].button("Prize Claims"):
        # st.info("Ask: 'What is the process for claiming prizes?'")