# Knowledge-Driven Marketing Assistant

The Knowledge-Driven Marketing Assistant is an AI-powered application that leverages a Vectorbased RAG system to unify knowledge retrieval and creative content generation. The platform ensures that both internal knowledge queries (FAQs, policy reference, employee support) and external creative marketing campaigns (copywriting, ad gen, content strategy) are powered by the same reliable information source.

## Features
**Ingestion Agent**: Parse and extract clean text from uploaded company documents 
**Indexing Agent**: Generate vector embeddings from text chunks and store them for similarity search.
**Retriever Agent**: Retrieve top relevant text chunks based on vector similarity to a user’s query.
**Task Classification Agent**: Analyze the user’s input to classify if the request is for a Q&A or Marketing content generation task.
**Q&A Agent**: Generate concise, citation-backed answers to knowledge queries using retrieved context
**Marketing Agent** :Generate creative marketing materials (ad copy, social posts, content calendar) 
**Logger Agent**: Record logs and metrics for each request to support debugging, auditing, and continuous improvement.

## Project Structure
![alt text][def]
[def]: image.png

## Setup
# Prerequisites
1. Install Ollama: Download and install from ollama.com
2. Pull a model: ollama pull llama3.2
3. Run a model: ollama run or ollama serve


# Project setup
1. **Clone the repo**
```bash
git clone <your repo>
cd circle-orchestrator-marketmentor
```
2. **Create the virtual environment**
```bash 
python -m venv venv
venv\Scripts\activate  #Activate environment
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Setup environment variable**
cp .env.example .env # edit the .env file to configure the Ollama setting

## Run the crew
```bash 
python main.py
```