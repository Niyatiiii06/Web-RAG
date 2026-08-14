from langchain_community.document_loaders import WebBaseLoader

urls = []
while True:
    url = input("Give the URL (or type 'no' to stop): ")
    if url.lower() == "no":
        break
    urls.append(url)

all_documents = []
for link in urls:
    loader = WebBaseLoader(link)
    url_docs = loader.load()
    for doc in url_docs:
        doc.metadata['source_url']= link
    all_documents.extend(url_docs)