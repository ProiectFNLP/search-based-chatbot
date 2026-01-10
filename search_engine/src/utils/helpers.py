import json
import os
import shutil
from typing import Generator, Optional

import fitz
import torch
from sentence_transformers import SentenceTransformer

from src.constants import TOP_K
from src.datasets.bm25_dataset import Bm25ChunkedDocumentDataset
from src.datasets.dense_dataset import DenseChunkedDocumentDataset
from src.datasets.tfidf_dataset import TfIdfChunkedDocumentDataset
from src.query.search_bm25 import search_bm25
from src.query.search_faiss import search_faiss
from src.query.search_tfidf import search_tfidf
from src.utils.cache import FileCache


def clear_cache_dir(cache_path: str = "articles"):
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)  # Delete the directory and all contents
        os.makedirs(cache_path)


def extract_text_from_pdf(contents: bytes) -> Generator[str, None, None]:
    """
    Extract text from a PDF file.
    :param contents: The PDF file contents to extract text from.
    :return: List with extracted text per page.
    """

    doc = fitz.open(stream=contents, filetype="pdf")
    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text()
        yield text

def extract_paragraphs_from_page(text: str) -> Generator[str, None, None]:
    """
    Extract paragraphs from a page.
    :param text: The page text to extract paragraphs from.
    :return: Generator yielding paragraphs from the page.
    """

    paragraphs = text.split("\n\n")
    for paragraph in paragraphs:
        if paragraph.strip():
            yield paragraph


def search_in_dataset(
    dataset: TfIdfChunkedDocumentDataset | DenseChunkedDocumentDataset | Bm25ChunkedDocumentDataset,
    search: str,
    file_cache: Optional[FileCache] = None  # FileCache to look up page/paragraph metadata
) -> Generator[str, None, None]:

    if isinstance(dataset, TfIdfChunkedDocumentDataset):
        generator = search_tfidf(search, dataset)
    elif isinstance(dataset, DenseChunkedDocumentDataset):
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        generator = search_faiss(search, dataset, model, TOP_K)
    elif isinstance(dataset, Bm25ChunkedDocumentDataset):
        generator = search_bm25(search, dataset)

    length = len(dataset)

    for paragraph, results in enumerate(generator):
        page_number = file_cache.get(f"paragraph_page_number:{paragraph}")
        data = {
            "results": [{
                "paragraph": int(result["id"]) + 1,
                "page": int(page_number) + 1,
                "score": float(result["score"])
            } for result in results],
            "current": paragraph,
            "total": length
        }
        yield f"data: {json.dumps(data)}\n\n"
