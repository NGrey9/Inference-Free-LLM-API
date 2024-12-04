# Import Libraries
import os
from dotenv import load_dotenv
load_dotenv()

from typing import Optional, Dict
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import uvicorn

from langchain_nvidia_ai_endpoints import ChatNVIDIA


# Main Source Code
NVIDIA_API_KEY = os.environ["NVIDIA_API_KEY"]


class InputMessage(BaseModel):
    message: str


class ChatAPI:
    def __init__(self):
        self.api  = FastAPI()
        self.client = ChatNVIDIA(
            model="meta/llama-3.1-405b-instruct",
            api_key=NVIDIA_API_KEY,
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024
        )
        self.history_chat = []
        self.setup_routes()


    def handle_message(self, input_message: str, memory_size: int = 20):
        """
        Handle user input message -> generate output message and return Response
        Params:
            input_message (str): user input message,
            memory_size (int): number of turn that the llm can remember in the conversation
        Returns:
            processed_message (dict): the generated output message by llm
        """
        try:
            processed_message = ""
            self.history_chat.append(
                {
                    "role": "user",
                    "content": input_message
                }
            )
            for chunk in self.client.stream(self.history_chat):
                processed_message += chunk.content
            self.history_chat.append(
                {
                    "role": "assistant",
                    "content": processed_message
                }
            )
            if len(self.history_chat) > (memory_size * 2):
                self.history_chat.pop(0)
                self.history_chat.pop(0)
            print(len(self.history_chat))
            return {"processed_message": processed_message}
        except Exception as e:
            return HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")


    def setup_routes(self):
        @self.api.post("/chat")
        async def handle_message(request: Request):
            input_data = await request.json()
            input_message = InputMessage(**input_data)
            return self.handle_message(input_message.message)

    def run(self, host='0.0.0.0', port=3000):
        uvicorn.run(self.api, host=host, port=port)


if __name__ == "__main__":
    api = ChatAPI()
    api.run()