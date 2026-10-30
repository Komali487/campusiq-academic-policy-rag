
"""
CampusIQ - Vector Store

Combines semantic similarity and policy-category
matching to improve academic policy retrieval.
"""

from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from query_understanding import analyze_query, POLICY_KEYWORDS


class PolicyVectorStore:
    """Semantic and category-aware policy search."""

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
            self.documents = []
            self.embeddings = None
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

    def calculate_category_bonus(
        self,
        text: str,
        categories: List[str]
    ) -> float:
        """Reward documents containing relevant policy terms."""

        text = text.lower()
        matched_terms = set()

        for category in categories:
            keywords = POLICY_KEYWORDS.get(category, [])

            for keyword in keywords:
                if keyword.lower() in text:
                    matched_terms.add(keyword.lower())

        return min(len(matched_terms) * 0.025, 0.12)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve documents using semantic and category scores."""

        if not self.documents or self.embeddings is None:
            return []

        query_info = analyze_query(query)

        query_embedding = self.model.encode(
            [query_info["cleaned_query"]],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        semantic_scores = np.dot(
            self.embeddings,
            query_embedding
        )

        results = []

        for index, document in enumerate(self.documents):
            result = document.copy()

            semantic_score = float(semantic_scores[index])

            category_bonus = self.calculate_category_bonus(
                document.get("text", ""),
                query_info["categories"]
            )

            result["score"] = semantic_score
            result["category_bonus"] = category_bonus
            result["rerank_score"] = (
                semantic_score + category_bonus
            )

            results.append(result)

        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return results[:top_k]


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
            f"\nSemantic score: {result['score']:.3f}"
            f"\nCategory bonus: {result['category_bonus']:.3f}"
            f"\nFinal ranking score: {result['rerank_score']:.3f}"
            f"\n{result['text']}"
        )
