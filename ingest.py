"""
ingest.py — Builds the FAISS vector index with full debug output.
Usage: python ingest.py
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import glob

DATA_DIR = "data"
INDEX_DIR = "faiss_index"

def ingest():
    # ── Step 1: Load each file manually with debug ────────────────────────────
    print("\n📂 Scanning data/ folder...")
    txt_files = glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
    txt_files += glob.glob(os.path.join(DATA_DIR, "*.txt"))
    txt_files = list(set(txt_files))  # deduplicate

    if not txt_files:
        print("❌ No .txt files found in data/! Check the folder.")
        return

    print(f"   Found {len(txt_files)} file(s):")
    for f in txt_files:
        size = os.path.getsize(f)
        print(f"   ✅ {f}  ({size} bytes)")

    # ── Step 2: Load documents ────────────────────────────────────────────────
    all_docs = []
    for filepath in txt_files:
        try:
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            # Tag each doc with its source filename for debugging
            for doc in docs:
                doc.metadata["source"] = os.path.basename(filepath)
            all_docs.extend(docs)
            print(f"\n   📄 Loaded: {filepath}")
            print(f"      First 150 chars: {docs[0].page_content[:150].strip()}")
        except Exception as e:
            print(f"   ❌ Failed to load {filepath}: {e}")

    print(f"\n   Total documents loaded: {len(all_docs)}")

    # ── Step 3: Split into chunks ─────────────────────────────────────────────
    print("\n✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\nNom:", "\n\n", "\n"],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"   Total chunks: {len(chunks)}")
    print("\n   Preview of all chunks:")
    for i, chunk in enumerate(chunks):
        src = chunk.metadata.get("source", "unknown")
        preview = chunk.page_content[:100].replace("\n", " ").strip()
        print(f"   [{i}] ({src}) {preview}...")

    # ── Step 4: Build embeddings & FAISS index ────────────────────────────────
    print("\n🔢 Building embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_DIR)
    print(f"\n✅ FAISS index saved to ./{INDEX_DIR}/")

    # ── Step 5: Quick retrieval test ──────────────────────────────────────────
    print("\n🔍 Quick retrieval test — 'employés atlas machinery':")
    results = db.similarity_search("employés atlas machinery", k=3)
    for i, r in enumerate(results):
        src = r.metadata.get("source", "?")
        print(f"   Result {i+1} ({src}): {r.page_content[:150].replace(chr(10), ' ').strip()}...")

    print("\n🔍 Quick retrieval test — 'employés noor location':")
    results = db.similarity_search("employés noor location", k=3)
    for i, r in enumerate(results):
        src = r.metadata.get("source", "?")
        print(f"   Result {i+1} ({src}): {r.page_content[:150].replace(chr(10), ' ').strip()}...")

    print("\n🎉 Done! Run: streamlit run app.py")

if __name__ == "__main__":
    ingest()