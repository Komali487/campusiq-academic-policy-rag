 # CampusIQ

## Context-Aware Academic Policy Reasoning and Decision Support System

CampusIQ is an AI-powered academic policy decision-support system that uses
Machine Learning, Natural Language Processing (NLP), and Retrieval-Augmented
Generation (RAG) to help students understand academic policies based on their
specific situations.

## Problem

Academic policies such as attendance requirements, examination eligibility,
medical exceptions, registration rules, scholarships, internships, and
deadlines are often distributed across multiple documents.

Students may therefore struggle to determine which policies apply to their
specific situation.

CampusIQ addresses this problem by understanding the student's context,
retrieving relevant policy evidence, and generating evidence-grounded guidance.

## Proposed Solution

The system follows this pipeline:

Student Query
↓
Query Understanding
↓
Intent & Situation Extraction
↓
Hybrid Retrieval
↓
Policy Knowledge Base
↓
Policy Relationship Analysis
↓
Evidence-Based RAG
↓
Verification
↓
Decision + Confidence
↓
Personalized Action Plan

## Key Features

- Context-aware academic policy retrieval
- ML-based query intent classification
- Student situation extraction
- Hybrid BM25 + semantic retrieval
- Vector database for policy knowledge
- Evidence-grounded RAG
- Policy relationship analysis
- Uncertainty detection
- Source/citation support
- Personalized action recommendations
- What-if academic scenario analysis

## Technologies

- Python
- Machine Learning
- NLP
- Retrieval-Augmented Generation (RAG)
- Sentence Embeddings
- BM25
- FAISS / Chroma
- Scikit-learn
- Streamlit
- GitHub

## Example

Student:

"I have 72% attendance because I was hospitalized.
Can I write the examination and what documents do I need?"

CampusIQ identifies:

- Attendance percentage
- Medical circumstance
- Examination context
- Possible policy exception

It then retrieves relevant policies and produces an evidence-grounded
response with applicable rules, supporting evidence, required actions,
and uncertainty information.

## Project Status

**Current status: Week 6 integration and testing completed; final validation in progress.**

CampusIQ is an academic prototype for context-aware academic policy question answering using Machine Learning, NLP, and Retrieval-Augmented Generation (RAG).

### Completed Work

- **Week 1:** Project proposal, problem definition, architecture, and data strategy.
- **Week 2:** Academic regulation PDF exploration and preprocessing.
- **Week 3:** Baseline machine learning experiment.
- **Week 4:** Model tuning and comparison experiments.
- **Week 5:** RAG pipeline development and related integration work.
- **Week 6:** Integrated pipeline execution, PDF processing, and integration test reporting.

### Current Dataset

- Source document: `R22B.Tech.AcademicRegulations.pdf`
- Extracted text chunks: 62
- Processed dataset: `data/processed/jntuh_policy_chunks.csv`

### Final Validation Checklist

- Verify integration test results and summary.
- Confirm retrieval and answer quality using representative academic policy questions.
- Review source citations and unsupported-answer handling.
- Ensure setup instructions and project documentation are accurate.

### Limitations

CampusIQ is an academic prototype. Its answers should be verified against official academic regulations, and it should not replace decisions made by the institution.


