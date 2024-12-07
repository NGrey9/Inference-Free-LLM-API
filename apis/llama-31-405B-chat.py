# Import Libraries
import os
from dotenv import load_dotenv
import langchain_core.messages
import langchain_core.messages.ai
load_dotenv()

from typing import Optional, Dict
from fastapi import (FastAPI, Request, HTTPException,
                     File, UploadFile, Form)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.messages.base import message_to_dict
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever

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
        self.api.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
        self.model = None
        self.setup_routes()
        self.vector_store = None
        self.file_status = False
        self.chat_history = []


    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            # ("system", "Answer the user's questions based on the context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        # self.chain = create_stuff_documents_chain(
        #     llm=self.model,
        #     prompt=prompt,
        #     output_parser=StrOutputParser()
        # )
        self.chain = prompt | self.model | StrOutputParser()


    def create_chain_file(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions based on the context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])
        document_chain = create_stuff_documents_chain(
            llm=self.model,
            prompt=prompt
        )
        retriever = self.vector_store.as_retriever(search_kwargs={"k":3})

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])

        history_aware_retriever = create_history_aware_retriever(
            llm=self.model,
            retriever=retriever,
            prompt=retriever_prompt
        )
        self.chain_file = create_retrieval_chain(history_aware_retriever, document_chain)



    def handle_message(self, message: str, chain, chain_type: str):        
        try:
            processed_message = chain.invoke({
                "chat_history": self.chat_history,
                "input": message
            })
            
            if chain_type == "chain_file":
                self.chat_history.append(HumanMessage(content=message))
                self.chat_history.append(AIMessage(content=processed_message["answer"]))
                return {"processed_message": processed_message["answer"]}
            else:
                self.chat_history.append(HumanMessage(content=message))
                self.chat_history.append(AIMessage(content=processed_message))
                return {"processed_message": processed_message}
            
        except Exception as e:
            raise e
            # return HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")
        
    def get_default_model(self):
        self.model = ChatNVIDIA(
                    model="meta/llama-3.1-405b-instruct",
                    api_key=NVIDIA_API_KEY,
                    temperature=0.2,
                    top_p=0.7,
                    max_tokens=1024
                )
        

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
                self.get_default_model()
                return {"error": "Model name not found!! Using default 'meta/llama-3.1-405b-instruct'",
                        "model_name": "meta/llama-3.1-405b-instruct"}
            
        @self.api.post("/chat")
        async def handle_message(file: Optional[UploadFile] = File(None),
                                 message: str = Form(...)):
            try:
                if self.model is None:
                    self.get_default_model()
                self.create_chain()
                if file:
                    if file.content_type != "application/pdf":
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid file type: {file.content_type}. Only PDF files are allowed."
                        )
                    with open(f"{ROOT_DIR}/uploaded/latest.pdf", "wb") as f:
                        content = await file.read()
                        f.write(content)
                    self.vector_store = create_vector_store()
                    self.create_chain_file()
                    self.file_status = True

                if self.file_status:
                    return self.handle_message(message, self.chain_file, chain_type="chain_file")
                else:
                    return self.handle_message(message, self.chain, chain_type="chain")
            except Exception as e:
                raise e
                return HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

    def run(self, host='0.0.0.0', port=3000):
        uvicorn.run(self.api, host=host, port=port)


if __name__ == "__main__":
    api = ChatAPI()
    api.run()