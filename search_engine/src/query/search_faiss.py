import pickle
from multiprocessing.managers import ListProxy
from multiprocessing.synchronize import Lock
from typing import Iterable, Optional, TypedDict, Generator
from multiprocessing import Pool, Manager

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from src.datasets.dense_dataset import DenseDocumentDataset, DenseBatchedOutput, DenseChunkedDocumentDataset
from src.datasets.utils import no_collate
from src.preprocessing.preprocess import preprocess_document

from sentence_transformers import SentenceTransformer

from src.utils.l2_normalizer import l2_normalize

from faiss import read_index
from src.constants import INDEX_PATH, TOP_K
from src.utils.redis_cache import FileCache


class SearchResult(TypedDict):
    document: str
    score: float
    id: int


def search_faiss_chunked(args: tuple[str, DenseChunkedDocumentDataset, int, list[SearchResult], Lock, SentenceTransformer, int]) -> list[SearchResult]:
    query = args[0]
    dataset = args[1]
    idx = args[2]
    global_results = args[3]
    lock = args[4]
    model = args[5]
    top_k = args[6]

    print(f"🔍 FAISS: Query = '{query}'")

    chunk = dataset.get_chunk(idx, model)

    documents = chunk['documents']
    #model = chunk['model']
    # faiss_index = read_index(INDEX_PATH)
    faiss_index = dataset.index

    chunk_size = len(documents)
    query = preprocess_document(query)
    print(f"🔍 FAISS: Preprocessed query = '{query}'")


    query_encoded = model.encode(
        [query],
        convert_to_numpy=True,
    )
    print(f"🔍 FAISS: Query encoded, shape = {query_encoded.shape}")

    normalized_query = l2_normalize(query_encoded)

    distances, ranked_indices = faiss_index.search(normalized_query, k=top_k)
    print(f"🔍 FAISS: Search complete - found {len(ranked_indices[0])} results")
    print(f"🔍 FAISS: Top distances = {distances[0][:min(3, len(distances[0]))]}")
    print(f"🔍 FAISS: Top indices = {ranked_indices[0][:min(3, len(ranked_indices[0]))]}")

    ranked_documents: list[SearchResult] = [{
        'id': idx*chunk_size + i,
        'document': documents[i],
        'score': distances[0][iteration]
    } for iteration, i in enumerate(ranked_indices[0])]

    print(f"🔍 FAISS: Created {len(ranked_documents)} ranked documents")
    print(f"🔍 FAISS: Top 3 ranked documents:")
    print(f"{'─'*80}")
    for i, doc in enumerate(ranked_documents[:3]):
        # Handle both bytes and string documents
        doc_text = doc['document']
        if isinstance(doc_text, bytes):
            doc_text = doc_text.decode('utf-8')
        print(f"\n[{i}] ID={doc['id']}, Score={doc['score']:.4f}")
        print(f"Full text:")
        print(doc_text)
        print(f"{'─'*80}")

    # 8. Lock
    with lock:
        global_results.extend(ranked_documents[:chunk_size])

        # 8. Sort by score
        tmp = sorted(global_results, key=lambda x: x['score'], reverse=True)[:chunk_size]
        global_results[:] = tmp

    return tmp

def search_faiss(query: str, dataset: DenseChunkedDocumentDataset, model: SentenceTransformer, top_k : int, file_cache: Optional[FileCache] = None) -> Generator[list[SearchResult], None, None]:
    if dataset.cache is not None and len(dataset.cache.subkeys()) > 0:
        multiprocessing = False
    else:
        multiprocessing = True

    multiprocessing = False

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
                 lock,
                 model,
                 top_k) for i in range(len(dataset))
            ]
            with Pool() as pool:
                for result in pool.imap_unordered(search_faiss_chunked, iterable):
                    all_results.extend(result)
        else:
            for i in range(len(dataset)):
                result = search_faiss_chunked((query, dataset, i, results, lock, model, top_k))
                all_results.extend(result)

    reranked = rerank_results(query, all_results, file_cache)
    yield reranked


def rerank_results(query: str, results: list[SearchResult], file_cache: Optional[FileCache] = None) -> list[SearchResult]:
    from src.utils.helpers import _get_cache_value

    print(f"\n{'='*60}")
    print(f"🔄 FAISS RERANKING: Starting")
    print(f"{'='*60}")
    print(f"🔄 RERANKING: Query = '{query}'")
    print(f"🔄 RERANKING: Total results to rerank = {len(results)}")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔄 RERANKING: Device = {device}")
    print(f"🔄 RERANKING: Loading CrossEncoder model...")
    reranking_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    print(f"🔄 RERANKING: ✓ CrossEncoder model loaded")

    valid_results = []
    pairs = []
    for result in results:
        paragraph = _get_cache_value(file_cache, f"paragraphs:{result['id']}")
        if paragraph:
            pairs.append((query, paragraph))
            valid_results.append(result)

    print(f"🔄 RERANKING: Valid results with paragraphs = {len(valid_results)}")
    print(f"🔄 RERANKING: Filtered out {len(results) - len(valid_results)} results without paragraphs")

    if not pairs:
        print(f"🔄 RERANKING: ⚠️ No valid pairs to rerank, returning original results")
        print(f"{'='*60}\n")
        return results

    print(f"🔄 RERANKING: Predicting scores for {len(pairs)} pairs...")
    scores = reranking_model.predict(pairs)
    print(f"🔄 RERANKING: ✓ Scores computed")
    print(f"🔄 RERANKING: Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    for i, result in enumerate(valid_results):
        old_score = result['score']
        result['score'] = float(scores[i])
        if i < 3:
            print(f"🔄 RERANKING: Result {i}: Old score={old_score:.4f} → New score={result['score']:.4f}")

    reranked = sorted(valid_results, key=lambda x: x['score'], reverse=True)
    print(f"🔄 RERANKING: ✓ Results sorted by new scores")
    print(f"🔄 RERANKING: Top 3 reranked scores: {[r['score'] for r in reranked[:3]]}")
    print(f"{'='*60}\n")

    return reranked
