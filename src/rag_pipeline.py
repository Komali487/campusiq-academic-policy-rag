"""
CampusIQ - RAG Pipeline
Retrieves academic policy evidence and generates grounded answers.
"""
import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from google import genai
from query_understanding import analyze_query
from vector_store import PolicyVectorStore
# Load the project-root .env file.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
class CampusIQRAG:
    """Retrieval-augmented academic policy assistant."""
    def __init__(self, model_name: str = "gemini-3.8-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        self.client = (
            genai.Client(api_key=api_key)
            if api_key
            else None
        )
        self.model_name = model_name
        self.vector_store = PolicyVectorStore()
    def load_documents(self, documents: List[Dict]) -> None:
        """Index academic policy chunks."""
        self.vector_store.add_documents(documents)
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve semantically relevant policy chunks."""
        query_info = analyze_query(query)
        return self.vector_store.search(
            query_info["cleaned_query"],
            top_k=top_k,
        )
    def build_context(self, documents: List[Dict]) -> str:
        """Build context and preserve source information."""
        return "\n\n---\n\n".join(
            f"Source: {doc.get('source', 'JNTUH regulations PDF')}\n"
            f"Chunk ID: {doc.get('chunk_id', 'Unknown')}\n"
            f"{doc.get('text', '')}"
            for doc in documents
        )
    def create_prompt(self, query: str, context: str) -> str:
        prompt = f"""
You are CampusIQ, an academic policy assistant.
Answer the student's question using ONLY the
retrieved academic policy evidence below.
STRICT RULES:
1. Answer ONLY what the student asked.
2. Do not include unrelated topics.
3. Do not combine different policy topics unless
   the question specifically asks about them.
4. Keep the answer within 2-4 sentences.
5. Include relevant regulation section numbers.
6. Never invent facts, marks, percentages,
   credits, fees, or dates.
7. If evidence is insufficient, clearly say so.
Student question:
{query}
Retrieved policy evidence:
{context}
Relevant answer:
"""
        return prompt.strip()
    def run(self, query: str, top_k: int = 5) -> Dict:

        """Retrieve evidence and generate a grounded answer."""
        query = query.strip()
        if not query:
            raise ValueError("Please enter an academic policy question.")
        query_analysis = analyze_query(query)
        documents = self.vector_store.search(
            query_analysis["cleaned_query"],
            top_k=top_k,
        )
        context = self.build_context(documents)
        prompt = self.create_prompt(query, context)
        result = {
            "query": query,
            "query_analysis": query_analysis,
            "retrieved_documents": documents,
            "context": context,
            "prompt": prompt,
            "answer": None,
            "answer_generation_model_used": False,
        }
        if not documents:
            result["answer"] = (
                "I could not find relevant information in the "
                "provided academic policy documents."
            )
            return result
        if self.client is None:
            result["answer"] = (
                "Gemini API key is missing. Add GEMINI_API_KEY "
                "to your project-root .env file."
            )
            return result
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            answer = response.text
            if not answer:
                answer = (
                    "Gemini did not return a text answer. "
                    "Please try again."
                )
            result["answer"] = answer
            result["answer_generation_model_used"] = True
        except Exception as exc:
            result["answer"] = (
                "Answer generation failed. Check your API key, "
                "model availability, API quota, and network connection. "
                f"Technical detail: {type(exc).__name__}"
            )
        return result
