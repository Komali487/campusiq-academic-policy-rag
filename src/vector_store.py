"""
CampusIQ - Vector Store

Creates semantic embeddings for academic policy chunks
and retrieves the most relevant chunks for a user query.
"""

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


class PolicyVectorStore:
    """Semantic search over academic policy document chunks."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Dict] = []
        self.embeddings = None

    def add_documents(
        self,
        documents: List[Dict]
    ) -> None:
        """Add documents and create their embeddings."""

        if not documents:
            return

        self.documents = documents

        texts = [
            document.get("text", "")
            for document in documents
        ]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Return the most semantically relevant documents."""

        if not self.documents or self.embeddings is None:
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []

        for index in top_indices:
            result = self.documents[index].copy()
            result["score"] = float(scores[index])
            results.append(result)

        return results


if __name__ == "__main__":

    sample_documents = [
        {
            "source": "attendance_policy.pdf",
            "chunk_id": 0,
            "text": (
                "Students must maintain the required "
                "attendance percentage to be eligible "
                "for semester examinations."
            ),
        },
        {
            "source": "fee_policy.pdf",
            "chunk_id": 0,
            "text": (
                "Students must pay tuition fees before "
                "the specified payment deadline."
            ),
        },
    ]

    store = PolicyVectorStore()

    store.add_documents(sample_documents)

    results = store.search(
        "How much attendance is required for exams?",
        top_k=2
    )

    for result in results:
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['score']:.3f}"
            f"\n{result['text']}"
        )
