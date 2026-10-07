"""
CampusIQ - Policy Retrieval

This module retrieves the most relevant document chunks
for a user's academic policy question.
"""

from typing import List, Dict
import re


def tokenize(text: str) -> List[str]:
    """Convert text into simple lowercase tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def calculate_keyword_score(query: str, document: str) -> float:
    """
    Calculate a simple relevance score based on
    the number of query words appearing in the document.
    """

    query_words = set(tokenize(query))
    document_words = set(tokenize(document))

    if not query_words:
        return 0.0

    matching_words = query_words.intersection(document_words)

    return len(matching_words) / len(query_words)


def retrieve_documents(
    query: str,
    documents: List[Dict],
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve the most relevant document chunks.

    Each document should contain:
        - source
        - chunk_id
        - text
    """

    scored_documents = []

    for document in documents:
        text = document.get("text", "")

        score = calculate_keyword_score(
            query,
            text
        )

        result = document.copy()
        result["score"] = score

        scored_documents.append(result)

    scored_documents.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored_documents[:top_k]


def format_retrieved_context(
    documents: List[Dict]
) -> str:
    """Combine retrieved chunks into context for the RAG model."""

    context_parts = []

    for document in documents:
        source = document.get("source", "Unknown")
        text = document.get("text", "")

        context_parts.append(
            f"Source: {source}\n{text}"
        )

    return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":

    sample_documents = [
        {
            "source": "attendance_policy.pdf",
            "chunk_id": 0,
            "text": (
                "Students must maintain a minimum "
                "attendance percentage to appear "
                "for semester examinations."
            ),
        },
        {
            "source": "fee_policy.pdf",
            "chunk_id": 0,
            "text": (
                "Tuition fees must be paid before "
                "the specified deadline."
            ),
        },
    ]

    question = "What attendance is required for exams?"

    results = retrieve_documents(
        question,
        sample_documents,
        top_k=2
    )

    print("Retrieved documents:")

    for result in results:
        print(
            f"\nSource: {result['source']}"
            f"\nScore: {result['score']:.2f}"
            f"\n{result['text']}"
        )
