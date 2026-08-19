import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.retrieval.vector_store import MedicalVectorStore


class MedicalHybridRetriever:
    def __init__(
        self,
        vector_store_dir: str = "data/processed/chroma_db",
        chunks_json_path: str = "data/processed/chunks.json",
        collection_name: str = "uspstf_guidelines"
    ):
        # 1. Load Vector Store
        self.vector_store = MedicalVectorStore(
            persist_directory=vector_store_dir,
            collection_name=collection_name
        )

        # 2. Load raw chunks and initialize BM25
        self.chunks_path = Path(chunks_json_path)
        if not self.chunks_path.exists():
            raise FileNotFoundError(f"Chunks file not found at: {self.chunks_path}")

        with open(self.chunks_path, "r", encoding="utf-8") as f:
            self.raw_chunks: List[Dict[str, Any]] = json.load(f)

        # Fast lookup mapping for O(1) metadata/text retrieval by ID
        self.chunk_lookup: Dict[str, Dict[str, Any]] = {
            str(chunk["id"]): chunk for chunk in self.raw_chunks
        }

        self.tokenized_corpus = [
            self._tokenize(chunk["text"]) for chunk in self.raw_chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        # Preserves hyphenated clinical terms (e.g., type-2, non-smoker) and alphanumeric codes
        return re.findall(r"\b[\w-]+\b", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        rrf_k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combines Dense Semantic Vector Search with Sparse BM25 Keyword Search
        using Weighted Reciprocal Rank Fusion (RRF).
        """
        fetch_limit = top_k * 4  # Expanded retrieval window for fusion stability

        # ----------------------------------------------------------------------
        # A. Dense Semantic Search (ChromaDB)
        # ----------------------------------------------------------------------
        query_emb = self.vector_store.get_embeddings([query])
        
        # Build vector search kwargs
        query_kwargs = {
            "query_embeddings": query_emb,
            "n_results": fetch_limit
        }
        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        vector_results = self.vector_store.collection.query(**query_kwargs)

        semantic_hits: Dict[str, Dict[str, Any]] = {}
        if vector_results.get("ids") and len(vector_results["ids"][0]) > 0:
            for rank, chunk_id in enumerate(vector_results["ids"][0]):
                cid_str = str(chunk_id)
                semantic_hits[cid_str] = {
                    "rank": rank + 1,
                    "text": vector_results["documents"][0][rank],
                    "metadata": vector_results["metadatas"][0][rank]
                }

        # ----------------------------------------------------------------------
        # B. Sparse Keyword Search (BM25)
        # ----------------------------------------------------------------------
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Filter by metadata FIRST, then drop zero-score keyword hits
        bm25_candidates = []
        for idx, score in enumerate(bm25_scores):
            if score <= 0.0:
                continue

            chunk = self.raw_chunks[idx]
            if filter_metadata:
                match = all(
                    chunk.get("metadata", {}).get(k) == v 
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            bm25_candidates.append((idx, score))

        # Rank valid BM25 candidates
        bm25_candidates.sort(key=lambda x: x[1], reverse=True)
        top_bm25 = bm25_candidates[:fetch_limit]

        bm25_hits: Dict[str, Dict[str, Any]] = {}
        for rank, (idx, _) in enumerate(top_bm25):
            chunk = self.raw_chunks[idx]
            cid_str = str(chunk["id"])
            bm25_hits[cid_str] = {
                "rank": rank + 1,
                "text": chunk["text"],
                "metadata": chunk.get("metadata", {})
            }

        # ----------------------------------------------------------------------
        # C. Weighted Reciprocal Rank Fusion (RRF)
        # ----------------------------------------------------------------------
        all_ids = set(semantic_hits.keys()).union(set(bm25_hits.keys()))
        combined_results = []

        for cid in all_ids:
            score = 0.0

            if cid in semantic_hits:
                score += alpha * (1.0 / (rrf_k + semantic_hits[cid]["rank"]))
            
            if cid in bm25_hits:
                score += (1.0 - alpha) * (1.0 / (rrf_k + bm25_hits[cid]["rank"]))

            # Retrieve text and metadata from direct hits or fast-lookup fallback
            doc_data = semantic_hits.get(cid) or bm25_hits.get(cid) or self.chunk_lookup.get(cid)

            combined_results.append({
                "id": cid,
                "score": score,
                "text": doc_data["text"] if doc_data else "",
                "metadata": doc_data["metadata"] if doc_data else {}
            })

        # Sort descending by fused RRF score
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:top_k]