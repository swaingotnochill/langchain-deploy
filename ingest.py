"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.faiss import FAISS

import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ.get('OPENAI_API_KEY')

def ingest_docs():
    """Get documents from web pages."""
    loader = CSVLoader(file_path="jsonSample50.csv", encoding="utf-8")
    data = loader.load()
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    docsearch = FAISS.from_documents(data, embeddings)

    # save docsearch(vectorstores)
    with open("docsearch.pkl", "wb") as f:
        pickle.dump(docsearch, f)

if __name__ == "__main__":
    ingest_docs()    