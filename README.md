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

🚧 Currently under development.

### Week 1
- Strategic project proposal
- Problem definition
- Literature review
- System architecture
- Data strategy
- ML/RAG methodology
- Evaluation strategy

### Upcoming
- Policy dataset collection
- Data preprocessing
- Query dataset creation
- Intent classification
- Semantic embeddings
- Hybrid retrieval
- RAG pipeline
- Streamlit application
- Evaluation

## Evaluation

The system will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Answer relevance
- Groundedness
- Citation correctness
- Unsupported-claim rate
- Uncertainty/refusal correctness

## Disclaimer

CampusIQ is an academic prototype and should not be treated as an
authoritative replacement for official institutional decisions.
