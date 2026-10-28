import tiktoken
def get_token(nodes):
# 1. Initialize the tokenizer (cl100k_base is the standard for modern models)
    enc = tiktoken.get_encoding("cl100k_base")
# 2. Select a node to inspect
    sample_node = nodes[70]
    node_text = sample_node.get_content()
# 3. Convert text to tokens (IDs)
    tokens = enc.encode(node_text)
 # 4. Visualization Output
    print(f"\n--- TOKEN ANALYSIS ---")
    print(f"Total Tokens in this Node: {len(tokens)}")
    print(f"First 10 Token IDs: {tokens[:5]}")

    # 5. Decode back to see the "Chunks of Characters"
    decoded_tokens = [enc.decode([t]) for t in tokens[:400]]
    print(f"First 10 Decoded Tokens: {decoded_tokens}")