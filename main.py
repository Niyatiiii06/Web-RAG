from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = []
while True:
    url = input("Give the URL (or type 'no' to stop): ")
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

print("\nTOTAL URLs:", len(urls))
print("TOTAL DOCUMENTS:", len(all_documents))

for i, doc in enumerate(all_documents, start=1):
    print(f"\n--- DOCUMENT {i} ---")
    print("SOURCE:", doc.metadata.get("source_url"))
    print("CHARACTERS:", len(doc.page_content))
    print(doc.page_content[:1000])