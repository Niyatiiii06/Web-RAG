import os

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

load_dotenv()


# -----------------------------------------
# LOAD WEB DOCUMENTS
# -----------------------------------------

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
        documents = loader.load()

        for doc in documents:
            doc.metadata["source_url"] = link

        all_documents.extend(documents)

        print(f"Loaded: {len(documents)} document(s)")

    except Exception as e:

        print(f"Could not load {link}")
        print(f"Error: {e}")


if not all_documents:
    print("No documents loaded.")
    exit()


# -----------------------------------------
# PARENT / CHILD SPLITTERS
# -----------------------------------------

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)


# -----------------------------------------
# EMBEDDINGS
# -----------------------------------------

emb = MistralAIEmbeddings(
    model="mistral-embed"
)


# -----------------------------------------
# VECTOR STORE + PARENT STORE
# -----------------------------------------

vectorstore = Chroma(
    collection_name="parent_child_demo",
    embedding_function=emb
)

store = InMemoryStore()


# -----------------------------------------
# PARENT DOCUMENT RETRIEVER
# -----------------------------------------

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)


# -----------------------------------------
# ADD ORIGINAL DOCUMENTS
# -----------------------------------------

retriever.add_documents(all_documents)


# -----------------------------------------
# TEST
# -----------------------------------------

query = input("\nQuestion: ")

results = retriever.invoke(query)

print(f"\nRetrieved parents: {len(results)}")


for i, doc in enumerate(results[:3], start=1):

    print(f"\n--- PARENT {i} ---")
    print("SOURCE:", doc.metadata.get("source_url"))
    print(doc.page_content[:1000])