import os
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.llama_parse import LlamaParse
from config import LLAMA_API_KEY # Ensure you have your LlamaCloud API key here

def load_documents(path):
    """
    Enhanced document loader using LlamaParse for complex layouts 
    and standard readers for simple files.
    """
    if not LLAMA_API_KEY:
        raise RuntimeError("Set LLAMA_CLOUD_API_KEY in your environment before loading documents.")

    try:
        # 1. Define LlamaParse for high-accuracy PDF parsing (handles tables/charts)
        # You can get a free API key at cloud.llamaindex.ai
        parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            result_type="markdown",  # Converts tables into readable MD format
            num_workers=1,           # Faster processing
            verbose=True,
            language="en",
            ignore_errors=False
        )

        # 2. Map file extensions to specialized parsers
        file_extractor = {".pdf": parser}

        # 3. Load data with metadata preservation
        # This ensures the AI knows exactly which file and page it is reading from
        reader = SimpleDirectoryReader(
            input_dir=path, 
            file_extractor=file_extractor,
            required_exts=[".pdf", ".docx", ".doc"],
            recursive=True
        )
        
        documents = reader.load_data()
        
        if not documents:
            print(f"Warning: No documents found in {path}")
            return None
            
        print(f"Successfully loaded and parsed {len(documents)} document pages.")
        return documents

    except Exception as e:
        print(f"Error during document loading: {str(e)}")
        return None