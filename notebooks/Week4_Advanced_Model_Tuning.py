# Week 4 - Advanced Model Tuning and Feature Engineering
# CampusIQ Academic Policy RAG

from pathlib import Path
import re
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("default")

# 1. Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "processed" / "jntuh_policy_label_review.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "week4_advanced_tuning_results.txt"
COMPARISON_PATH = OUTPUT_DIR / "week4_model_comparison.csv"

# 2. Load manual labels
df = pd.read_csv(DATA_PATH)
df["text"] = df["text"].fillna("").astype(str).str.strip()
df["verified_category"] = (
    df["verified_category"].fillna("").astype(str).str.strip()
)
df = df[
    (df["text"] != "") & (df["verified_category"] != "")
].copy()

print("Original labeled examples:", len(df))
print("\nOriginal category counts:")
print(df["verified_category"].value_counts().to_string())

# Exclude categories with fewer than 3 examples so 3-fold stratified
# cross-validation can be attempted.
counts = df["verified_category"].value_counts()
eligible_categories = counts[counts >= 3].index
excluded = counts[counts < 3]
df = df[df["verified_category"].isin(eligible_categories)].copy()

if df["verified_category"].nunique() < 2:
    raise SystemExit("Need at least two categories with 3 or more examples.")

print("\nExamples used:", len(df))
print("Categories used:", sorted(df["verified_category"].unique()))
print("Excluded categories:", excluded.to_dict())

# 3. Feature engineering
# These numeric features supplement word and character TF-IDF.
def count_terms(text, terms):
    text = text.lower()
    return sum(
        len(re.findall(r"\b" + re.escape(term) + r"\b", text))
        for term in terms
    )

def add_features(frame):
    frame = frame.copy()
    text = frame["text"].astype(str)

    frame["char_count"] = text.str.len()
    frame["word_count"] = text.apply(
        lambda value: len(re.findall(r"\b\w+\b", value))
    )
    frame["sentence_count"] = text.apply(
        lambda value: max(1, len(re.findall(r"[.!?]+", value)))
    )
    frame["uppercase_ratio"] = text.apply(
        lambda value: (
            sum(char.isupper() for char in value) / max(len(value), 1)
        )
    )
    frame["digit_ratio"] = text.apply(
        lambda value: (
            sum(char.isdigit() for char in value) / max(len(value), 1)
        )
    )

    term_groups = {
        "attendance_terms": ["attendance", "attend", "absent"],
        "exam_terms": ["examination", "examinations", "exam", "exams"],
        "credit_terms": ["credit", "credits", "grade", "grades", "marks"],
        "eligibility_terms": ["promotion", "eligibility", "detained"],
        "fee_terms": ["fee", "fees", "tuition"],
        "semester_terms": ["semester", "semesters"],
    }

    for feature, terms in term_groups.items():
        frame[feature] = text.apply(lambda value: count_terms(value, terms))

    return frame

df = add_features(df)

numeric_features = [
    "char_count",
    "word_count",
    "sentence_count",
    "uppercase_ratio",
    "digit_ratio",
    "attendance_terms",
    "exam_terms",
    "credit_terms",
    "eligibility_terms",
    "fee_terms",
    "semester_terms",
]

print("\nEngineered numeric features:", numeric_features)

# 4. Train/test split
X = df[["text"] + numeric_features]
y = df["verified_category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nTraining examples:", len(X_train))
print("Test examples:", len(X_test))

# 5. Simple majority-class benchmark
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_predictions = dummy.predict(X_test)

# 6. Original-style word TF-IDF baseline
baseline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

baseline.fit(X_train["text"], y_train)
baseline_predictions = baseline.predict(X_test["text"])

# 7. Advanced feature preprocessing
# Word TF-IDF captures words and phrases.
# Character TF-IDF captures partial words and spelling variations.
# StandardScaler scales numerical features.
preprocessor = ColumnTransformer(
    transformers=[
        (
            "word_tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
            "text",
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                max_features=3000,
                sublinear_tf=True,
            ),
            "text",
        ),
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
    ],
    remainder="drop",
)

# 8. Hyperparameter tuning with 3-fold cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

experiments = {
    "Tuned Logistic Regression": (
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    ),
    "Tuned Linear SVM": (
        LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
        {
            "classifier__C": [0.1, 1.0, 10.0],
        },
    ),
}

results = []
best_estimators = {}

def record_result(name, y_true, predictions, best_params=None, cv_score=None):
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, predictions, average="weighted", zero_division=0
        ),
        "best_cv_macro_f1": cv_score,
        "best_parameters": str(best_params or {}),
    })

record_result(
    "Majority baseline",
    y_test,
    dummy_predictions,
)
record_result(
    "Word TF-IDF Logistic Regression baseline",
    y_test,
    baseline_predictions,
)

for name, (classifier, param_grid) in experiments.items():
    print(f"\nTuning: {name}")

    pipeline = Pipeline([
        ("features", preprocessor),
        ("classifier", classifier),
    ])

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    predictions = search.predict(X_test)
    best_estimators[name] = search.best_estimator_

    print("Best parameters:", search.best_params_)
    print("Best cross-validation macro F1:", round(search.best_score_, 3))
    print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
    print(
        "Test macro F1:",
        round(f1_score(y_test, predictions, average="macro", zero_division=0), 3),
    )

    record_result(
        name,
        y_test,
        predictions,
        best_params=search.best_params_,
        cv_score=search.best_score_,
    )

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

# 9. Save comparison table
comparison = pd.DataFrame(results)
comparison.to_csv(COMPARISON_PATH, index=False)

# 10. Save detailed report
with open(RESULTS_PATH, "w", encoding="utf-8") as f:
    f.write("CampusIQ - Week 4 Advanced Model Tuning\n")
    f.write("=" * 48 + "\n")
    f.write("Input: manually assigned categories in label review CSV\n")
    f.write("Labels are provisional and have not been independently verified.\n\n")
    f.write(f"Eligible examples: {len(df)}\n")
    f.write(f"Training examples: {len(X_train)}\n")
    f.write(f"Test examples: {len(X_test)}\n")
    f.write(f"Excluded categories: {excluded.to_dict()}\n")
    f.write(f"Numeric features: {numeric_features}\n\n")
    f.write("Experiment comparison:\n")
    f.write(comparison.to_string(index=False))
    f.write("\n\nInterpretation and limitations:\n")
    f.write(
        "- Compare test metrics on the same held-out split.\n"
        "- GridSearchCV uses training data only; the test set is reserved "
        "for final comparison.\n"
        "- The feature pipeline combines word TF-IDF, character TF-IDF, "
        "and scaled numeric features.\n"
        "- Only C was tuned for Logistic Regression and Linear SVM.\n"
        "- The dataset is small, so both cross-validation and test metrics "
        "may vary substantially with different samples.\n"
        "- Labels are provisional; scores do not establish reliable "
        "classification of real-world policy text.\n"
        "- Scaling is applied to numeric features. TF-IDF features are "
        "already normalized by the vectorizer by default.\n"
    )

print("\nModel comparison:")
print(comparison.to_string(index=False))
print("\nResults saved to:", RESULTS_PATH)
print("Comparison CSV saved to:", COMPARISON_PATH)
