from multiprocessing import Pool, Manager
from typing import Generator, Optional
from multiprocessing.synchronize import Lock
from src.datasets.bm25_dataset import Bm25ChunkedDocumentDataset
from src.preprocessing.preprocess import preprocess_document
from src.query.search_tfidf import SearchResult
import numpy as np
import torch
from sentence_transformers import CrossEncoder

from src.utils.redis_cache import FileCache



def search_bm25_chunked(args: tuple[str, Bm25ChunkedDocumentDataset, int, list[SearchResult], Lock]) -> list[SearchResult]:
    query = args[0]
    dataset = args[1]
    idx = args[2]
    global_results = args[3]
    lock = args[4]


    chunk = dataset[idx]
    bm25 = chunk['bm25']
    documents = chunk['documents']
    chunk_size = len(documents)
    query = preprocess_document(query)

    query_tokens = query.split()

    # 6. Compute document scores
    scores = bm25.get_scores(query_tokens)

    # 7. Rank sentences
    ranked_indices = np.argsort(scores)[::-1]

    ranked_documents: list[SearchResult] = [{
        'id': idx*chunk_size + i,
        'document': documents[i],
        'score': scores[i]
    } for i in ranked_indices]
    # 8. Lock
    with lock:
        global_results.extend(ranked_documents[:chunk_size])

        # 8. Sort by score
        tmp = sorted(global_results, key=lambda x: x['score'], reverse=True)[:chunk_size]
        global_results[:] = tmp
    return tmp


def search_bm25(query: str, dataset: Bm25ChunkedDocumentDataset, file_cache: Optional[FileCache] = None) -> Generator[list[SearchResult], None, None]:

    if dataset.cache is not None and len(dataset.cache.subkeys()) > 0:
        multiprocessing = False
    else:
        multiprocessing = True

    multiprocessing = False

    all_results = []
    with Manager() as manager:
        results = manager.list()
        lock = manager.Lock()
        if multiprocessing:
            iterable = [
                (query,
                 dataset,
                 i,
                 results,
                 lock) for i in range(len(dataset))
            ]
            with Pool() as pool:
                for result in pool.imap_unordered(search_bm25_chunked, iterable):
                    all_results.extend(result)
        else:
            for i in range(len(dataset)):
                result = search_bm25_chunked((query, dataset, i, results, lock))
                all_results.extend(result)

    reranked = rerank_results(query, all_results, file_cache)
    yield reranked


def rerank_results(query: str, results: list[SearchResult], file_cache: Optional[FileCache] = None) -> list[SearchResult]:
    from src.utils.helpers import _get_cache_value

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)

    valid_results = []
    pairs = []
    for result in results:
        paragraph = _get_cache_value(file_cache, f"paragraphs:{result['id']}")
        if paragraph:
            pairs.append((query, paragraph))
            valid_results.append(result)

    if not pairs:
        return results

    scores = reranking_model.predict(pairs)

    for i, result in enumerate(valid_results):
        result['score'] = float(scores[i])

    return sorted(valid_results, key=lambda x: x['score'], reverse=True)
