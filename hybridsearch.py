import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
# 1. GET URLs
urls = []
while True:
    url = input("Give the URL (or type 'no' to stop): ").strip()
    if url.lower() == "no":
        break
    if not url:
        print("URL cannot be empty.")
        continue
    urls.append(url)


# 2. LOAD WEB DOCUMENTS
all_documents = []
for link in urls:
    print(f"\nLoading: {link}")
    try:
        loader = WebBaseLoader(link)
        url_docs = loader.load()
        for doc in url_docs:
            doc.metadata["source_url"] = link
        all_documents.extend(url_docs)
        print(f"Loaded: {len(url_docs)} document(s)")
    except Exception as e:
        print(f"Could not load {link}")
        print(f"Error: {e}")


print("\n-----------------------------")
print("TOTAL URLs:", len(urls))
print("TOTAL DOCUMENTS:", len(all_documents))
print("-----------------------------")

if not all_documents:
    print("No documents were loaded.")
    exit()


# 3. CHUNK DOCUMENTS
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(all_documents)
print("TOTAL CHUNKS:", len(chunks))

if not chunks:
    print("No chunks were created.")
    exit()


# 4. CREATE DENSE EMBEDDINGS
emb = MistralAIEmbeddings(
    model="mistral-embed"
)


# 5. CREATE CHROMA VECTOR STORE
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    persist_directory="chroma_db"
)
print("Chroma vector store ready.")


# 6. CREATE BM25 RETRIEVER
bm25 = BM25Retriever.from_documents(chunks)
print("BM25 retriever ready.")


# 7. RRF FUSION
def rrf_fusion(dense_results, bm25_results, k=60):
    scores = {}
    documents = {}
    # Dense results
    for rank, doc in enumerate(dense_results, start=1):
        key = doc.page_content
        documents[key] = doc
        scores[key] = (
            scores.get(key, 0)
            + 1 / (k + rank))
        
    # BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        key = doc.page_content
        documents[key] = doc
        scores[key] = (
            scores.get(key, 0)
            + 1 / (k + rank)
        )
    # Sort by combined RRF score
    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )
    return [documents[key] for key in ranked_keys]


# 8. RUN HYBRID RETRIEVAL
query = input("\nHybrid query: ")

# Dense retrieval
dense_results = vectorstore.similarity_search(
    query,
    k=5
)

# BM25 retrieval
bm25_results = bm25.invoke(query)

# Combine using RRF
hybrid_results = rrf_fusion(
    dense_results,
    bm25_results
)


# 9. DISPLAY RESULTS
for i, doc in enumerate(hybrid_results[:5], start=1):
    print(f"\n--- HYBRID RESULT {i} ---")
    print(
        "SOURCE:",
        doc.metadata.get("source_url", "Unknown")
    )
    print(
        doc.page_content[:500]
    )

context_parts = []

for doc in hybrid_results[:5]:
    source = doc.metadata.get("source_url", "Unknown")

    context_parts.append(
        f"Source: {source}\n"
        f"Content:\n{doc.page_content}"
    )

context = "\n\n---\n\n".join(context_parts)

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)

messages = [
    SystemMessage(
        content=(
            "Answer the question using only the provided context. "
            "If the context does not contain the answer, say so. "
            "Do not invent information."
        )
    ),
    HumanMessage(
        content=(
            f"Context:\n\n{context}\n\n"
            f"Question: {query}"
        )
    )
]

response = llm.invoke(messages)

print("\n--- ANSWER ---")
print(response.content)