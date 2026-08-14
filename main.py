import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

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

splitter= RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200
)
chunks= splitter.split_documents(all_documents)

emb= MistralAIEmbeddings(model="mistral-embed")

vectorstore= Chroma.from_documents(
    documents= chunks,
    embedding= emb,
    persist_directory= 'chroma_db'
)
llm= ChatMistralAI(model='mistral-small-latest')

while True:
    query= input('\nAsk a question (or type exit): ')
    if query.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    context=[]
    results= vectorstore.similarity_search(query, k=3)

    for doc in results:
        source= doc.metadata.get('source_url', 'Unknown')
        context.append( 
            f'SOURCE: {source}'
            f'CONTENT: {doc.page_content}'    )
    context= '\n\n---\n\n'.join(context)

    messages=[
        SystemMessage(content=("You are a web RAG assistant. "
                "Answer the question using only the provided context. "
                "If the answer is not present in the context, say "
                "'I couldn't find that in the provided webpages.' "
                "Do not invent information.")),
        HumanMessage(content=(f"Context:\n\n{context}\n\n"
                f"Question: {query}"))
    ]
    respone= llm.invoke(messages)
    print(respone.content)

