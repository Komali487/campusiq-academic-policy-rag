"""
CampusIQ - Evaluation

Utilities for evaluating the retrieval performance
of the academic policy RAG system.
"""

from typing import List, Dict


def calculate_retrieval_accuracy(
    results: List[Dict],
    expected_source: str
) -> float:
    """
    Check whether the expected policy document
    appears in the retrieved results.
    """

    if not results:
        return 0.0

    for result in results:
        if result.get("source") == expected_source:
            return 1.0

    return 0.0


def evaluate_queries(
    test_cases: List[Dict],
    retriever
) -> Dict:
    """
    Evaluate multiple test questions.

    Each test case should contain:
        - query
        - expected_source
    """

    scores = []

    for test_case in test_cases:
        query = test_case["query"]
        expected_source = test_case["expected_source"]

        results = retriever(query)

        score = calculate_retrieval_accuracy(
            results,
            expected_source
        )

        scores.append(score)

    if not scores:
        return {
            "accuracy": 0.0,
            "total_questions": 0,
        }

    accuracy = sum(scores) / len(scores)

    return {
        "accuracy": accuracy,
        "total_questions": len(scores),
        "correct": int(sum(scores)),
    }


if __name__ == "__main__":

    sample_results = [
        {
            "source": "attendance_policy.pdf",
            "chunk_id": 0,
            "score": 0.80,
            "text": "Minimum attendance is required.",
        }
    ]

    accuracy = calculate_retrieval_accuracy(
        sample_results,
        "attendance_policy.pdf"
    )

    print(f"Retrieval accuracy: {accuracy:.2f}")
