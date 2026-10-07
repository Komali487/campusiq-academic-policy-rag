"""
CampusIQ - RAG Pipeline

Connects query understanding, document retrieval,
and answer generation into one pipeline.
"""

from typing import List, Dict

from query_understanding import analyze_query
from retrieval import retrieve_documents, format_retrieved_context


def build_prompt(query: str, context: str) -> str:
    """
    Build a grounded prompt using retrieved policy information.
    """

    return f"""
You are CampusIQ, an academic policy assistant.

Answer the student's question using ONLY the policy
information provided in the context below.

If the answer cannot be found in the context,
clearly say that the information is not available
in the provided academic policy documents.

Do not invent rules, percentages, dates, fees,
or other policy information.

Student question:
{query}

Policy context:
{context}

Answer:
""".strip()


def generate_answer(
    query: str,
    retrieved_documents: List[Dict]
) -> str:
    """
    Generate a simple grounded answer.

    This version prepares the answer from retrieved
    policy context. An LLM can be connected later.
    """

    if not retrieved_documents:
        return (
            "I could not find relevant information in "
            "the available academic policy documents."
        )

    relevant_documents = [
        document
        for document in retrieved_documents
        if document.get("score", 0) > 0
    ]

    if not relevant_documents:
        return (
            "I could not find a sufficiently relevant "
            "policy section to answer this question."
        )

    context = format_retrieved_context(relevant_documents)

    prompt = build_prompt(query, context)

    return prompt


def run_rag_pipeline(
    query: str,
    documents: List[Dict],
    top_k: int = 5
) -> Dict:
    """
    Run the complete CampusIQ RAG pipeline.
    """

    query_info = analyze_query(query)

    retrieved_documents = retrieve_documents(
        query_info["cleaned_query"],
        documents,
        top_k=top_k
    )

    answer = generate_answer(
        query,
        retrieved_documents
    )

    return {
        "query": query,
        "query_analysis": query_info,
        "retrieved_documents": retrieved_documents,
        "answer": answer,
    }


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

    question = "What attendance is required for exams?"

    result = run_rag_pipeline(
        question,
        sample_documents
    )

    print(result["answer"])
