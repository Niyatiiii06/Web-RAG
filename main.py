from langchain_community.document_loaders import WebBaseLoader

url = "https://www.ibm.com/think/insights/artificial-intelligence-future"
loader = WebBaseLoader(url)
url_docs = loader.load()

print("Documents loaded:", len(url_docs))

print("\nCONTENT:")
print(url_docs[0].page_content[:500])

print("\nMETADATA:")
print(url_docs[0].metadata)