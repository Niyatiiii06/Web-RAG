import os

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


# --------------------------------------------------
# 1. LOAD WEB DOCUMENTS
# --------------------------------------------------

urls = []

while True:

    url = input("Give the URL (or type 'no' to stop): ").strip()

    if url.lower() == "no":
        break

    if not url:
        print("URL cannot be empty.")
        continue

    urls.append(url)


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


if not all_documents:
    print("No documents were loaded.")
    exit()


# --------------------------------------------------
# 2. CHUNK DOCUMENTS
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(all_documents)

print(f"\nTotal documents: {len(all_documents)}")
print(f"Total chunks: {len(chunks)}")


# --------------------------------------------------
# 3. EMBEDDINGS
# --------------------------------------------------

embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)


# --------------------------------------------------
# 4. VECTOR STORE
# --------------------------------------------------

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="query_rewrite_chroma"
)

print("Vector store ready.")


# --------------------------------------------------
# 5. LLM
# --------------------------------------------------

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0
)


# --------------------------------------------------
# 6. QUERY REWRITING
# --------------------------------------------------

def rewrite_query(query):

    messages = [
        SystemMessage(
            content=(
                "You are a search query rewriting assistant. "
                "Rewrite the user's question into a concise search query "
                "that will help retrieve relevant information from a "
                "document collection. "
                "Preserve the important concepts and technical terms. "
                "Return only the rewritten search query. "
                "Do not answer the question."
            )
        ),
        HumanMessage(
            content=query
        )
    ]

    response = llm.invoke(messages)

    return response.content.strip()


# --------------------------------------------------
# 7. ASK A QUESTION
# --------------------------------------------------

query = input("\nAsk a question: ")

rewritten_query = rewrite_query(query)


print("\n-----------------------------")
print("ORIGINAL QUERY:")
print(query)

print("\nREWRITTEN QUERY:")
print(rewritten_query)
print("-----------------------------")


# --------------------------------------------------
# 8. RETRIEVE USING REWRITTEN QUERY
# --------------------------------------------------

results = vectorstore.similarity_search(
    rewritten_query,
    k=3
)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------

for i, doc in enumerate(results, start=1):

    print(f"\n--- RESULT {i} ---")

    print(
        "SOURCE:",
        doc.metadata.get(
            "source_url",
            "Unknown"
        )
    )

    print(
        doc.page_content[:1000]
    )