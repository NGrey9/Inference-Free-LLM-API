import os
from dotenv import load_dotenv
load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings


ROOT_DIR = os.environ["ROOT_DIR"]

def create_vector_store():
    try:
        pdf_filepath = f"{ROOT_DIR}/uploaded/latest.pdf"

        loader = PyPDFLoader(pdf_filepath)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20
        )
        embedding = OllamaEmbeddings(model="llama3")

        docs = loader.load()
        splitted_docs = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(docs, embedding=embedding)
        return vector_store

    except Exception as e:
        raise e