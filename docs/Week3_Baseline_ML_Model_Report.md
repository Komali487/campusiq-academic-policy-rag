# Week 3 - Baseline Machine Learning Model
## CampusIQ Academic Policy RAG

### 1. Objective
Build an initial text-classification baseline using academic-regulation text chunks.

### 2. Dataset
- Source: R22B.Tech.AcademicRegulations.pdf
- Total chunks: 62
- Features: source, chunk_id, text
- Missing values in the verified dataset: 0

### 3. Important Label Limitation
The original dataset did not contain ground-truth category labels. Categories were assigned using keyword-based rules for this prototype. These labels are heuristic labels, not manually verified policy categories. Therefore, evaluation measures agreement with these generated labels and does not establish real-world policy classification accuracy.

### 4. Preprocessing and Split
- Text representation: TF-IDF
- Train/test split: 75% / 25%, with random_state=42 and stratification
- Test examples: 16
- Categories with fewer than two examples were excluded from the modeling subset.

### 5. Models
1. Majority-class DummyClassifier baseline
2. TF-IDF with Logistic Regression (max_iter=1000, class_weight="balanced")

Logistic Regression was selected as a simple, interpretable baseline for sparse text features.

### 6. Results
- Majority-class baseline accuracy: 50%
- TF-IDF + Logistic Regression accuracy: 75%
- Logistic Regression macro-average F1-score: 0.55
- Logistic Regression weighted-average F1-score: 0.73

### 7. Interpretation
The TF-IDF + Logistic Regression model performed better than the majority-class baseline on this single test split. However, the dataset is small, category distribution is imbalanced, and labels were generated from keyword rules. The scores should not be interpreted as validated accuracy on real academic-policy categories.

### 8. Limitations and Future Work
- Create a larger dataset with manually verified category labels.
- Use stratified cross-validation where class counts permit.
- Evaluate on unseen, human-labeled policy chunks.
- Consider adding more policy documents and comparing additional models.

### 9. Files
- `notebooks/Week3_Baseline_ML_Model.py`
- `data/processed/jntuh_policy_chunks.csv`
- `data/processed/week3_baseline_results.txt`
