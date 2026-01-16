import asyncio
import json
import multiprocessing
import os
import pickle
import time
import uuid
import random
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from multiprocessing import Pool
from typing import Literal, Optional

from llm_api import generate_response, generate_summary

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from settings import settings

from src.constants import CHUNK_SIZE, MESSAGE_CONTEXT_WINDOW
from src.datasets.dense_dataset import DenseChunkedDocumentDataset, DenseFileDocumentDataset
from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset, TfIdfFileDocumentDataset
from src.dependencies import get_file_cache, init_dependencies
from src.preprocessing.preprocess import preprocess_document
from src.preprocessing.string_list_utils import preprocess_string_list
from src.utils import pool_executor
from src.utils.helpers import (
    extract_text_from_pdf,
    extract_paragraphs_from_page,
    search_in_dataset,
)
from src.utils.redis_cache import make_hash, FileCache
from src.utils.redis import init_redis, close_redis

from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset
from src.datasets.dense_dataset import DenseChunkedDocumentDataset
from src.datasets.bm25_dataset import Bm25ChunkedDocumentDataset, Bm25FileDocumentDataset


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis(app)
    init_dependencies()
    pool_executor.executor = ProcessPoolExecutor()
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(pool_executor.executor, pool_executor.__init__worker) for _ in range(10)]
    yield
    pool_executor.executor.shutdown(wait=True)
    await close_redis()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def preprocess_and_cache(file_cache: FileCache, key: str, pool_executor) -> None:
    if file_cache.exists(f"preprocessed:{key}"):
        return

    document = file_cache.get(f"documents:{key}")
    if document is None:
        raise ValueError(f"Document {key} not found in cache.")
    document = document.decode('utf-8')

    preprocessed = await asyncio.get_running_loop().run_in_executor(
        pool_executor.executor,
        preprocess_document,
        document
    )
    file_cache.set(f"preprocessed:{key}", preprocessed)


async def prepare_pdf_cache(session_id: str, file_cache: FileCache):
    """Retrieve and prepare the PDF cache for a given session.

    This function encapsulates the steps to look up the file hash from the
    session id, obtain the `pdf` subcache, validate the stored length, and
    ensure any missing pages are preprocessed and cached.

    Returns:
        tuple(pdf_cache, file_hash, length)
    Raises:
        HTTPException(404) if session or length not found
    """

    file_hash = file_cache.get(f"session:{session_id}")
    file_hash = file_hash.decode('utf-8') if file_hash else None

    if not file_hash:
        raise HTTPException(status_code=404, detail="Session not found.")

    pdf_cache = file_cache.subcache(f"pdf:{file_hash}")
    length = pdf_cache.get("length")
    if length is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    length = int(length)

    existing_preprocessed = set(pdf_cache.subkeys("preprocessed"))
    print(existing_preprocessed)
    missing_preprocessed = [str(i) for i in range(length) if str(i) not in existing_preprocessed]
    print(missing_preprocessed)
    start = time.time()

    loop = asyncio.get_event_loop()

    await asyncio.gather(*[
        preprocess_and_cache(pdf_cache, key, pool_executor)
        for key in missing_preprocessed
    ])

    end = time.time()
    print(f"Preprocessing took {end - start:.2f} seconds")


    return pdf_cache, file_hash, length


@app.post("/upload")
async def upload(file: UploadFile = File(...), file_cache: FileCache = Depends(get_file_cache)):
    # clear_cache_dir(BASE_CACHE_DIR)
    if file.filename is None or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_hash = await make_hash(file)

    contents = await file.read()
    content_pages = list(extract_text_from_pdf(contents))
    pdf_cache = file_cache.subcache(f"pdf:{file_hash}")
    print(pdf_cache)
    print(file_cache)

    paragraph_index = 0
    for page_num, page in enumerate(content_pages):
        paragraphs = list(extract_paragraphs_from_page(page))
        pdf_cache.set(f"documents:{page_num}", page)
        for j, paragraph in enumerate(paragraphs):
            pdf_cache.set(f"paragraphs:{j}", paragraph)
        paragraph_index += len(paragraphs)

    pdf_cache.set("no_pages", str(len(content_pages)))
    pdf_cache.set("length", str(paragraph_index))
    # preprocessed_pages = preprocess_string_list(content_pages)

    session_id = str(uuid.uuid4())

    file_cache.set(f"session:{session_id}", file_hash)

    return {"session_id": session_id}


async def _search(session_id: str, search: str, file_cache: FileCache, mode: Literal['tfidf', 'faiss', 'bm25']):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    print(f"Received search request for session {session_id} with search string: {search}")

    pdf_cache, file_hash, length = await prepare_pdf_cache(session_id, file_cache)

    if mode == 'tfidf':
        base_dataset = TfIdfFileDocumentDataset(pdf_cache, length=length)
        cache = file_cache.subcache(f"tfidf:{file_hash}")
        dataset = TfIdfChunkedDocumentDataset(base_dataset, chunk_size=CHUNK_SIZE, cache=cache)
    elif mode == 'faiss':
        base_dataset = DenseFileDocumentDataset(pdf_cache, length=length)
        cache = file_cache.subcache(f"faiss:{file_hash}")
        dataset = DenseChunkedDocumentDataset(base_dataset, chunk_size=CHUNK_SIZE, cache=cache)
    elif mode == 'bm25':
        base_dataset = Bm25FileDocumentDataset(pdf_cache, length=length)
        cache = file_cache.subcache(f"bm25:{file_hash}")
        dataset = Bm25ChunkedDocumentDataset(base_dataset, chunk_size=CHUNK_SIZE, cache=cache)
    else:
        raise HTTPException(status_code=400, detail="Invalid mode. Use 'tfidf' or 'faiss'.")

    results = search_in_dataset(dataset, search, file_cache)

    return results

@app.get("/search-tf-idf")
async def search_tfidf(session_id: str, search: str, file_cache: FileCache = Depends(get_file_cache)):
    """
    Search for a string in a PDF file.
    :param file: The PDF file to search in.
    :param search: The string to search for.
    :param file_cache: The file cache to use for storing and retrieving the PDF.
    :return: A JSON response with the number of pages in the PDF.
    """

    results = await _search(session_id, search, file_cache, mode='tfidf')
    return StreamingResponse(results, media_type="text/event-stream")

@app.get("/search-faiss")
async def search_faiss(session_id: str, search: str, file_cache: FileCache = Depends(get_file_cache)):
    """
    Search for a string in a PDF file using FAISS.
    :param session_id: The session ID for the uploaded PDF.
    :param search: The string to search for.
    :param file_cache: The file cache to use for storing and retrieving the PDF.
    :return: A JSON response with the search results.
    """

    results = await _search(session_id, search, file_cache, mode='faiss')
    return StreamingResponse(results, media_type="text/event-stream")

@app.get("/search-bm25")
async def search_bm25(session_id: str, search: str, file_cache: FileCache = Depends(get_file_cache)):
    """
    Search for a string in a PDF file using BM25.
    :param session_id: The session ID for the uploaded PDF.
    :param search: The string to search for.
    :return: A JSON response with the search results.
    """

    results = await _search(session_id, search, file_cache, mode='bm25')
    return StreamingResponse(results, media_type="text/event-stream")


# TODO: make fake generator from responses
@app.get("/send-prompt")
async def send_prompt(
    session_id: str,
    prompt: str,
    search_mode: Literal['tfidf', 'faiss', 'bm25'] = 'faiss',
    llm_model: Literal['llama', 'gpt-4o-mini'] = 'llama',
    file_cache: FileCache = Depends(get_file_cache)
):
    """
    Send a prompt to the chatbot and get a response.
    The prompt and response are stored in conversation history.

    :param session_id: The session ID for the uploaded PDF.
    :param prompt: The user's question/prompt.
    :param search_mode: Which search method to use for context retrieval.
    :param file_cache: The file cache for storing conversation history.
    :return: A JSON response with the bot's answer and conversation history.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    print(f"Received prompt for session {session_id}: {prompt}")

    pdf_cache, file_hash, length = await prepare_pdf_cache(session_id, file_cache)

    # Get conversation history and concatenate with prompt
    conversation_summary = pdf_cache.get(f"conversation_summary:{session_id}")
    if conversation_summary is None:
        conversation_summary = ""

    context_messages = []
    for i in range(MESSAGE_CONTEXT_WINDOW):
        message = pdf_cache.get(f"messages:{length - i - 1}")
        if message is not None:
            context_messages.append(message)

    # Process query (see if we need to do coreference resolution)
    search_query = prompt

    # Search the document for relevant context
    print(search_query, search_mode)
    results = await _search(session_id, search_query, file_cache, mode=search_mode) #intoarce un generator

    # Generate response using LLM
    response = generate_response(results, prompt)
    print("Generated response:", response)

    # Save summary of conversation
    conversation_summary = generate_summary(prompt, response, conversation_summary)
    pdf_cache.set(f"conversation_summary:{session_id}", conversation_summary)

    # Return response
    async def generator():
        yield f"data: {json.dumps({'response': response})}\n\n"
        await asyncio.sleep(1)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

@app.get("/send-message-dummy")
async def send_message(session_id: str, message: str, file_cache: FileCache = Depends(get_file_cache)):

    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")

    print(f"Received message for session {session_id}: {message}")

    pdf_cache, file_hash, length = await prepare_pdf_cache(session_id, file_cache)

    results = simulate_chat_response()

    # Modify this
    return StreamingResponse(results, media_type="text/event-stream")



def simulate_chat_response():
    responses = [
        "Sure, I can help with that. ",
        "Let me check the document for you. ",
        "Here is the information you requested. ",
        "Is there anything else you would like to know? ",
        "Thank you for using our service! "
    ]

    for response in responses:
        sleep_time = random.uniform(0.5, 1.5)
        time.sleep(sleep_time)  # Simulate delay
        data = {
            "response": response
        }
        yield f"data: {json.dumps(data)}\n\n"
