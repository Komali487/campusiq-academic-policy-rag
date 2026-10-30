"""
CampusIQ - Query Understanding

Cleans student questions and detects relevant academic policy categories.
"""

import re
from typing import Dict, List


POLICY_KEYWORDS = {
    "attendance": [
        "attendance",
        "attend",
        "absent",
        "absence",
        "shortage",
        "percentage",
        "condonation",
        "condone",
        "present",
    ],
    "examination": [
        "exam",
        "examination",
        "semester end",
        "end examination",
        "internal",
        "supplementary",
        "backlog",
        "reappear",
        "eligibility",
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
        "absence due to illness",
    ],
    "grading": [
        "grade",
        "grading",
        "cgpa",
        "gpa",
        "marks",
        "credits",
        "credit points",

    ],
    "academic_calendar": [
        "academic calendar",
        "semester dates",
        "start date",
        "end date",
        "working days",
        "academic year",
        "holidays",
    ],
    "promotion": [
        "promotion",
        "promoted",
        "detained",
        "next semester",
        "credits required",
    ],
}


def clean_query(query: str) -> str:
    """Clean and normalize a user query."""

    query = query.strip().lower()
    query = re.sub(r"[^\w\s?%-]", " ", query)
    query = re.sub(r"\s+", " ", query)

    return query


def detect_policy_categories(query: str) -> List[str]:
    """Identify likely policy categories using word boundaries."""

    categories = []

    for category, keywords in POLICY_KEYWORDS.items():
        for keyword in keywords:
            pattern = r"(?<!\w)" + re.escape(keyword) + r"(?!\w)"
            if re.search(pattern, query):
                categories.append(category)
                break

    return categories


def analyze_query(query: str) -> Dict:
    """Return normalized query and detected policy categories."""

    cleaned_query = clean_query(query)
    categories = detect_policy_categories(cleaned_query)

    return {
        "original_query": query,
        "cleaned_query": cleaned_query,
        "categories": categories,
        "has_category": bool(categories),
    }


if __name__ == "__main__":
    test_queries = [
        "What is the minimum attendance required for exams?",
        "How many marks are required to pass?",
        "What is the fee for condonation?",
        "Can I get promoted to the next semester?",
    ]

    for question in test_queries:
        print(f"\nQuestion: {question}")
        print(analyze_query(question))
