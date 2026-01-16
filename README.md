# Search-based Chatbot

# Setup
 - Create a virtual environment with Python 3.10:
   `python3.10 -m venv .venv`
- Activate it
- Install requirements from backend:
  `pip install -r search_engine/requirements.txt`
- Install redis server:
  * For Linux:
  
  `sudo apt update`
  
  `sudo apt install redis-server -y`

  * For BREW:

  `brew install redis`

  `brew services start redis`

- Download Ollama from here: `https://ollama.com/download`

  Run Ollama: `ollama run llama3.2`

# How to run:

## Frontend
`cd frontend`,
`npm run dev`

## Backend
`cd search_engine`,
`uvicorn main:app --reload`

# How to use
1. Upload a pdf document and wait for it to be loaded
2. Select a search model (Tf-Idf, Bm25, or Faiss)
3. Enter a search query
4. Press "Search" and wait for the whole document to be analyzed
5. Press any of the search results to jump to the respective page
