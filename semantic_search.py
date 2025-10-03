import spacy
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# --- Step 1: Load a pre-trained spaCy model ---
# You might need to download a spaCy model first:
# python -m spacy download en_core_web_md
# or for Arabic: python -m spacy download ar_core_news_md

try:
    nlp = spacy.load("en_core_web_md") # Using a medium English model for demonstration
    print("spaCy model loaded successfully.")
except OSError:
    print("Downloading spaCy model (en_core_web_md). This may take a moment...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    print("spaCy model downloaded and loaded.")

# --- Step 2: Prepare documents and generate embeddings ---
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, agile fox leaps over a sluggish canine.",
    "Artificial intelligence is transforming industries.",
    "Machine learning is a subset of AI.",
    "Deep learning is a specialized form of machine learning.",
    "The cat sat on the mat.",
    "Dogs are loyal companions."
]

print("Generating document embeddings...")
document_embeddings = []
for doc_text in documents:
    doc = nlp(doc_text)
    if doc.has_vector:
        document_embeddings.append(doc.vector)
    else:
        print(f"Warning: Document \'{doc_text}\' has no vector. Skipping.")

if not document_embeddings:
    print("Error: No document embeddings could be generated. Ensure spaCy model has vectors.")
    exit()

document_embeddings = np.array(document_embeddings).astype("float32")

# --- Step 3: Create a Faiss index ---
dimension = document_embeddings.shape[1] # Dimension of the embeddings
index = faiss.IndexFlatL2(dimension) # Using L2 distance for similarity
index.add(document_embeddings)
print(f"Faiss index created with {index.ntotal} documents.")

# --- Step 4: Perform a semantic search query ---
def semantic_search(query_text, k=3):
    query_doc = nlp(query_text)
    if not query_doc.has_vector:
        print("Error: Query has no vector. Cannot perform search.")
        return []

    query_embedding = np.array([query_doc.vector]).astype("float32")

    # Search the Faiss index
    distances, indices = index.search(query_embedding, k) # k nearest neighbors

    print(f"\nSemantic search for: \'{query_text}\'")
    results = []
    for i in range(k):
        doc_index = indices[0][i]
        if doc_index < len(documents):
            # Calculate cosine similarity for better interpretability (Faiss gives L2 distance)
            sim = cosine_similarity(query_embedding, [document_embeddings[doc_index]])[0][0]
            results.append({
                "document": documents[doc_index],
                "similarity": sim,
                "distance": distances[0][i]
            })
    return results

if __name__ == "__main__":
    search_queries = [
        "Tell me about animals.",
        "What is AI?",
        "Tell me about learning algorithms."
    ]

    for query in search_queries:
        results = semantic_search(query, k=2)
        for res in results:
            print(f"  - Document: \'{res["document"]}\' (Similarity: {res["similarity"]:.4f})")

    print("\nSemantic search project setup complete. Run `python semantic_search.py` to execute.")
    print("Remember to install spaCy model if not already present: `python -m spacy download en_core_web_md`")

