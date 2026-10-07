"""
CampusIQ - Query Understanding

This module prepares user questions for academic policy retrieval.
"""

import re
from typing import Dict, List


POLICY_KEYWORDS = {
    "attendance": [
        "attendance",
        "absent",
        "absence",
        "shortage",
        "percentage",
    ],
    "examination": [
        "exam",
        "examination",
        "semester",
        "internal",
        "marks",
        "supplementary",
    ],
    "fees": [
        "fee",
        "fees",
        "payment",
        "tuition",
        "refund",
    ],
    "leave": [
        "leave",
        "permission",
        "medical leave",
        "holiday",
    ],
    "grading": [
        "grade",
        "grading",
        "cgpa",
        "gpa",
        "marks",
        "credits",
    ],
    "academic_calendar": [
        "academic calendar",
        "semester dates",
        "start date",
        "end date",
    ],
}


def clean_query(query: str) -> str:
    """Clean and normalize a user query."""

    query = query.strip().lower()

    # Remove unnecessary punctuation.
    query = re.sub(r"[^\w\s?%-]", " ", query)

    # Remove repeated spaces.
    query = re.sub(r"\s+", " ", query)

    return query


def detect_policy_categories(query: str) -> List[str]:
    """Identify likely policy categories from a user query."""

    categories = []

    for category, keywords in POLICY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                categories.append(category)
                break

    return categories


def analyze_query(query: str) -> Dict:
    """
    Analyze a user question and return structured information
    for the retrieval system.
    """

    cleaned_query = clean_query(query)

    categories = detect_policy_categories(cleaned_query)

    return {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "categories": categories,
        "has_category": bool(categories),
    }


if __name__ == "__main__":
    example_query = "What is the minimum attendance required for exams?"

    result = analyze_query(example_query)

    print("Query analysis:")
    print(result)
