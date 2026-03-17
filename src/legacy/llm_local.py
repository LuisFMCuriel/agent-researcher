from langchain_ollama import ChatOllama, OllamaEmbeddings


def demo_chat():
    llm = ChatOllama(model="llama3.2:3b", temperature=0.2)
    msg = llm.invoke("In one sentence, define RAG (Retrieval-Augmented Generation) in the context of LLM applications.")
    print("\n=== CHAT RESULT ===\n")
    print(msg.content)


def demo_embeddings():
    emb = OllamaEmbeddings(model="nomic-embed-text")
    v = emb.embed_query("experiment 43: unet baseline, dice 0.81")
    print("\n=== EMBEDDING RESULT ===\n")
    print("vector length:", len(v))
    print("first 8 dims:", v[:8])


if __name__ == "__main__":
    demo_chat()
    demo_embeddings()