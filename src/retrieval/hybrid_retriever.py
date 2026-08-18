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

        self.tokenized_corpus = [
            self._tokenize(chunk["text"]) for chunk in self.raw_chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Combines Semantic Vector Search with BM25 Keyword Search using Reciprocal Rank Fusion (RRF).
        """
        # A. Semantic Search
        query_emb = self.vector_store.get_embeddings([query])
        
        vector_results = self.vector_store.collection.query(
            query_embeddings=query_emb,
            n_results=top_k * 2,
            where=filter_metadata
        )

        semantic_hits = {}
        if vector_results["ids"] and len(vector_results["ids"][0]) > 0:
            for rank, chunk_id in enumerate(vector_results["ids"][0]):
                semantic_hits[str(chunk_id)] = {
                    "rank": rank + 1,
                    "text": vector_results["documents"][0][rank],
                    "metadata": vector_results["metadatas"][0][rank]
                }

        # B. BM25 Keyword Search
        tokenized_query = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: top_k * 2]

        bm25_hits = {}
        for rank, idx in enumerate(top_bm25_indices):
            chunk = self.raw_chunks[idx]
            
            if filter_metadata:
                match = all(chunk["metadata"].get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue
                    
            bm25_hits[str(chunk["id"])] = {
                "rank": rank + 1,
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }

        # C. Fusion (RRF)
        all_ids = set(semantic_hits.keys()).union(set(bm25_hits.keys()))
        combined_scores = []
        k_constant = 60

        for cid in all_ids:
            score = 0.0
            doc_data = None

            if cid in semantic_hits:
                score += alpha * (1.0 / (k_constant + semantic_hits[cid]["rank"]))
                doc_data = semantic_hits[cid]

            if cid in bm25_hits:
                score += (1.0 - alpha) * (1.0 / (k_constant + bm25_hits[cid]["rank"]))
                if not doc_data:
                    doc_data = bm25_hits[cid]

            combined_scores.append({
                "id": cid,
                "score": score,
                "text": doc_data["text"],
                "metadata": doc_data["metadata"]
            })

        combined_scores.sort(key=lambda x: x["score"], reverse=True)
        return combined_scores[:top_k]