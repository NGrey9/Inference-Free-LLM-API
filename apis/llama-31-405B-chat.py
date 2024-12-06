# Import Libraries
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Optional, Dict
from fastapi import (FastAPI, Request, HTTPException,
                     File, UploadFile, Form)
from pydantic import BaseModel
import uvicorn

from langchain_core.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# from utils import create_vector_store``


# Main Source Code
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]
ROOT_DIR = os.environ["ROOT_DIR"]


class ModelName(BaseModel):
    model_name: str
    
def create_vector_store():
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores.faiss import FAISS
        from langchain_ollama.embeddings import OllamaEmbeddings

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

class ChatAPI:
    def __init__(self):
        self.api  = FastAPI()
        self.model = None
        self.setup_routes()
        self.vector_store = None
        self.file_status = False
        self.chat_history = []

    def create_chain(self, vector_store):
        # prompt = ChatPromptTemplate.from_messages([("human", "{input_message}")])
        prompt = ChatPromptTemplate.from_template("""
        Answer the user's question.
        Context: {context}
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompt,
            output_parser=StrOutputParser()
        )
        retriever = vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain



    def handle_message(self, input: str):
        """
        Handle user input message -> generate output message and return Response
        Params:
            input (str): user input message,
            model_name (str): llm model name that you want to inference
        Returns:
            processed_message (dict): the generated output message by llm
        """
        
        try:
            prompt = ChatPromptTemplate.from_messages(
                [("human", "{input")])
            

            parser = StrOutputParser()
            
            chain = prompt | self.model | parser

            processed_message = chain.invoke({
                "input": input
            })

            return {"processed_message": processed_message}
        except Exception as e:
            return HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        

    def setup_routes(self):
        @self.api.post("/model")
        async def handle_model_choosing(request: Request):
            try:
                data = await request.json()
                data_model = ModelName(**data)
                self.model = ChatNVIDIA(
                    model=data_model.model_name,
                    api_key=NVIDIA_API_KEY,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024
                )
                return {"model_name": data_model.model_name}
            except Exception as e:
                self.model = ChatNVIDIA(
                    model="meta/llama-3.1-405b-instruct",
                    api_key=NVIDIA_API_KEY,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024
                )
                return {"error": "Model name not found!! Using default 'meta/llama-3.1-405b-instruct'",
                        "model_name": "meta/llama-3.1-405b-instruct"}
            
        @self.api.post("/chat")
        async def handle_message(file: Optional[UploadFile] = File(None),
                                 message: str = Form(...)):
            if self.model is None:
                self.model = ChatNVIDIA(
                    model="meta/llama-3.1-405b-instruct",
                    api_key=NVIDIA_API_KEY,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024
                )
            if file:
                if file.content_type != "application/pdf":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid file type: {file.content_type}. Only PDF files are allowed."
                    )
                with open(f"{ROOT_DIR}/uploaded/latest.pdf", "wb") as f:
                    content = await file.read()
                    f.write(content)
                self.file_status = True
            if self.file_status:
                self.vector_store = create_vector_store()
                retrieval_chain = self.create_chain(self.vector_store)
                processed_message = retrieval_chain.invoke({
                    "input": message
                })
                return {"processed_message": processed_message["answer"]}
            else:
                return self.handle_message(message)

    def run(self, host='0.0.0.0', port=3000):
        uvicorn.run(self.api, host=host, port=port)


if __name__ == "__main__":
    api = ChatAPI()
    api.run()