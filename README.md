# Deploy Personal LLM Chatbot Using Free API

This repository is a simple applycation which built for deploying an LLM chatbot to website. This application has the ability to allow users to import .pdf file, then ask and answer about that file. In addition, this chatbot can also remember the chat history in each session, so it take advantage of the information that user provided in previous messages to answer latest question most accurately.

## Getting Started

First, you need to folk this repository to your machine:
```sh
$ git clone https://github.com/NGrey9/Inference-Free-LLM-API
```

My simple application used free API from NVIDIA NIM for chatbot (meta/llama-3.1-405b-instruct and meta/llama-3.1-70b-instruct in this repository). You can get free API key at  <a href="https://build.nvidia.com/explore/discover">this link</a>, and you can choose any model. So you can get your free API key by clicking `Get API Key` button. After that, you need to create .env file and pass ``NVIDIA_API_KEY = "<your_API_key>"`` into .env file

## Install Dependencies

Make sure your machine has `python>=3.10` installed and to install necessary packages you can use:
```sh
$ pip install -r requirements.txt
```

## Start Python Back-end API For Website

Use above command to start back-end API:
```sh
$ python apis/llama-31-405B-chat.py
```

Then, you can start front-end server by using:
```sh
$ python deploy_website.py
```

The python back-end API will run on 3000 port and you can enter website at <a href="http://localhost:8088">8088 port</a>

## Result 

The RAG system will be deployed on your CPU. So, It take quite many time retrieving vector data. However, this application is actually useful and worth trying.
<p align="center">
  <img src="https://github.com/NGrey9/Inference-Free-LLM-API/tree/api/assets/chat.gif">
</p>

It also remember the history of the conversation.
<p align="center">
  <img src="https://github.com/NGrey9/Inference-Free-LLM-API/tree/api/assets/chat1.gif">
</p>

So what about Vietnamese...
<p align="center">
  <img src="https://github.com/NGrey9/Inference-Free-LLM-API/tree/api/assets/chat2.gif">
</p>

## Provisional Conclusion
This is a simple LLM chatbot application with web UI for QA task. I will update the Vision QA feature or cool stuff soon. Thank you for your following