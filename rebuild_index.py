"""
Rebuild the Chroma vector index for the Welch digital twin.

Run this whenever the embedding model changes (e.g. Google retired
text-embedding-004 -> gemini-embedding-001). The embedding model used here MUST
match `GoogleGenAIEmbedding(model_name=...)` in app1.py, because vectors from
different models are not comparable.

Usage:
    GOOGLE_API_KEY=...  python rebuild_index.py
    (or put GOOGLE_API_KEY in a local .env)
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings as LlamaSettings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# --- config: keep in sync with app1.py ---
EMBED_MODEL = "gemini-embedding-001"
EXPERT_NAME = "Stewart Cary Welch"
PERSIST_DIR = "./chroma_db"
DOCS_DIR = "./documents"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise SystemExit("Set GOOGLE_API_KEY (env or .env) before rebuilding.")

collection_name = EXPERT_NAME.replace(" ", "_").lower() + "_collection"

# Fresh rebuild — the old vectors use a different (retired) model and are
# incompatible, so wipe the directory first.
if Path(PERSIST_DIR).exists():
    print(f"Removing old index at {PERSIST_DIR} ...")
    shutil.rmtree(PERSIST_DIR)

embed_model = GoogleGenAIEmbedding(model_name=EMBED_MODEL, api_key=api_key)
LlamaSettings.embed_model = embed_model
LlamaSettings.chunk_size = CHUNK_SIZE
LlamaSettings.chunk_overlap = CHUNK_OVERLAP

print(f"Loading documents from {DOCS_DIR} ...")
documents = SimpleDirectoryReader(
    DOCS_DIR, recursive=True, required_exts=[".pdf", ".txt", ".md"]
).load_data()
print(f"  loaded {len(documents)} document(s)")

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(name=collection_name)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

print(f"Embedding with {EMBED_MODEL} and building index ...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context,
    transformations=[splitter], show_progress=True,
)
print(f"  collection '{collection_name}' now holds {collection.count()} chunks")

# Smoke-test retrieval (no LLM needed) through the freshly built index.
nodes = index.as_retriever(similarity_top_k=3).retrieve(
    "Who was Stewart Cary Welch and what did he collect?"
)
print("\n--- retrieval smoke test ---")
print(f"retrieved {len(nodes)} chunks; top score {nodes[0].score:.3f}")
print(nodes[0].node.text[:300].replace("\n", " "))
print("\n✅ Rebuild complete.")
