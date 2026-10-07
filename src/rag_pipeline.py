 """
CampusIQ - RAG Pipeline

Connects query understanding, semantic retrieval,
and grounded answer generation.
"""

from typing import List, Dict

from query_understanding import analyze_query
from vector_store import PolicyVectorStore


class CampusIQRAG:
    """Main RAG system for CampusIQ."""

    def __init__(self):
        self.vector_store = PolicyVectorStore()

    def load_documents(
        self,
        documents: List[Dict]
    ) -> None:
        """Load policy documents into the semantic search index."""

        self.vector_store.add_documents(documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Retrieve the most relevant policy chunks."""

        query_info = analyze_query(query)

        return self.vector_store.search(
            query_info["cleaned_query"],
            top_k=top_k
        )

    def build_context(
        self,
        documents: List[Dict]
    ) -> str:
        """Build context from retrieved policy chunks."""

        if not documents:
            return "No relevant policy information was found."

        context_parts = []

        for document in documents:
            source = document.get(
                "source",
                "Unknown source"
            )

            text = document.get("text", "")

            context_parts.append(
                f"Source: {source}\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def create_prompt(
        self,
        query: str,
        context: str
    ) -> str:
        """Create a grounded prompt for the answer generator."""

        return f"""
You are CampusIQ, an academic policy assistant.

Answer the student's question using ONLY the
provided academic policy context.

Do not invent policies, dates, fees, percentages,
deadlines, or other information.

If the answer is not available in the context,
say that the information is not available in
the provided policy documents.

Student question:
{query}

Academic policy context:
{context}

Answer:
""".strip()

    def run(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """Run the complete retrieval pipeline."""

        query_analysis = analyze_query(query)

        documents = self.vector_store.search(
            query_analysis["cleaned_query"],
            top_k=top_k
        )

        context = self.build_context(documents)

        prompt = self.create_prompt(
            query,
            context
        )

        return {
            "query": query,
            "query_analysis": query_analysis,
            "retrieved_documents": documents,
            "context": context,
            "prompt": prompt,
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

    rag = CampusIQRAG()

    rag.load_documents(sample_documents)

    result = rag.run(
        "What attendance is required for exams?"
    )

    print("Retrieved policy information:\n")

    for document in result["retrieved_documents"]:
        print(
            f"Source: {document['source']}\n"
            f"Score: {document['score']:.3f}\n"
            f"{document['text']}\n"
        )

    print("\nGenerated prompt:\n")
    print(result["prompt"])
