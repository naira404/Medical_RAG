import json
from pathlib import Path
from typing import List, Dict, Any
from fastembed import TextEmbedding
import chromadb


class MedicalVectorStore:
    def __init__(
        self,
        persist_directory: str = "data/processed/chroma_db",
        collection_name: str = "uspstf_guidelines",
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading lightweight embedding model: {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """توليد التضمينات لأي قائمة نصوص أو أسئلة باستخدام FastEmbed."""
        return [emb.tolist() for emb in self.model.embed(texts)]

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64):
        total_chunks = len(chunks)
        print(f"Indexing {total_chunks} chunks into ChromaDB...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            
            ids = [str(item["id"]) for item in batch]
            documents = [item["text"] for item in batch]
            
            metadatas = []
            for item in batch:
                meta = item["metadata"].copy()
                if "pages" in meta and isinstance(meta["pages"], list):
                    meta["pages"] = ",".join(map(str, meta["pages"]))
                if meta.get("publication_year") is None:
                    meta["publication_year"] = 0
                metadatas.append(meta)

            embeddings = self.get_embeddings(documents)

            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            print(f"Indexed {min(i + batch_size, total_chunks)}/{total_chunks} chunks...")

        print("Indexing completed successfully!")