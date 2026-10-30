
"""CampusIQ Streamlit application."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from data_processing import process_all_pdfs
from rag_pipeline import CampusIQRAG


st.set_page_config(
    page_title="CampusIQ - Academic Policy Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 CampusIQ")
st.subheader("Academic Policy Assistant")
st.write(
    "Ask questions about academic regulations. "
    "Answers are grounded in the policy PDF available to the system."
)


@st.cache_resource(show_spinner="Loading academic policy documents...")
def load_rag():
    documents = process_all_pdfs()

    rag = CampusIQRAG()
    rag.load_documents(documents)

    return rag, len(documents)


try:
    rag, document_count = load_rag()
except Exception as exc:
    st.error(
        "Could not load the academic policy documents. "
        f"Error type: {type(exc).__name__}"
    )
    st.stop()


if document_count == 0:
    st.error(
        "No policy chunks were found. Check that your PDF "
        "is available in data/raw/."
    )
    st.stop()

st.caption(f"Policy chunks loaded: {document_count}")

question = st.text_area(
    "Enter your academic policy question",
    placeholder="Example: What are the attendance requirements for exams?",
    height=100,
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Searching policy documents and preparing an answer..."):
            try:
                result = rag.run(question.strip(), top_k=5)
            except Exception as exc:
                st.error(
                    "The request could not be completed. "
                    f"Error type: {type(exc).__name__}"
                )
                st.stop()

        st.markdown("### Answer")
        st.write(result.get("answer") or "No answer was generated.")

        if not result.get("answer_generation_model_used"):
            st.info(
                "A Gemini-generated answer was not confirmed. "
                "Check the message above and your API configuration."
            )

        with st.expander("View retrieved policy evidence", expanded=False):
            evidence = result.get("retrieved_documents", [])

            if not evidence:
                st.write("No relevant policy chunks were retrieved.")
            else:
                for index, doc in enumerate(evidence, start=1):
                    source = doc.get("source", "Unknown source")
                    chunk_id = doc.get("chunk_id", "Unknown")
                    score = doc.get("score")

                    st.markdown(
                        f"**{index}. {source} — Chunk {chunk_id}**"
                    )

                    if score is not None:
                        st.caption(f"Retrieval similarity score: {score:.3f}")

                    st.write(doc.get("text", ""))
                    st.divider()

st.caption(
    "Please verify important academic decisions against the "
    "official regulations and the relevant college authority."
)
