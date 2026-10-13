# Week 4 - Baseline Model Using Manual Labels
# CampusIQ Academic Policy RAG

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# 1. Locate project files
project_dir = Path(__file__).resolve().parent.parent
labels_path = (
    project_dir / "data" / "processed" / "jntuh_policy_label_review.csv"
)
results_path = (
    project_dir / "data" / "processed" / "week4_verified_baseline_results.txt"
)

# 2. Load manually assigned labels
df = pd.read_csv(labels_path)
df["verified_category"] = (
    df["verified_category"].fillna("").astype(str).str.strip()
)
df["text"] = df["text"].fillna("").astype(str).str.strip()

df = df[
    (df["text"] != "") & (df["verified_category"] != "")
].copy()

print("Dataset shape:", df.shape)
print("\nManual label counts:")
print(df["verified_category"].value_counts().to_string())

# 3. Exclude classes with fewer than 2 examples.
# At least 2 examples are needed for a stratified split.
counts = df["verified_category"].value_counts()
eligible_categories = counts[counts >= 2].index

excluded_categories = counts[counts < 2]
eligible = df[df["verified_category"].isin(eligible_categories)].copy()

if len(eligible) < 10 or eligible["verified_category"].nunique() < 2:
    raise SystemExit("Not enough eligible labeled examples to train.")

X = eligible["text"]
y = eligible["verified_category"]

# A 25% test split; stratification preserves class proportions where possible.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print("\nExcluded categories with fewer than 2 examples:")
if excluded_categories.empty:
    print("None")
else:
    print(excluded_categories.to_string())

print("\nTraining examples:", len(X_train))
print("Test examples:", len(X_test))
print("Categories evaluated:", sorted(y.unique()))

# 4. Majority-class baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
dummy_predictions = dummy.predict(X_test)

# 5. TF-IDF + Logistic Regression
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 6. Evaluation
labels = sorted(y.unique())

baseline_accuracy = accuracy_score(y_test, dummy_predictions)
model_accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(
    y_test, predictions, average="macro", zero_division=0
)
weighted_f1 = f1_score(
    y_test, predictions, average="weighted", zero_division=0
)

report = classification_report(
    y_test,
    predictions,
    labels=labels,
    zero_division=0,
)
matrix = confusion_matrix(y_test, predictions, labels=labels)

print("\nMajority baseline accuracy:", round(baseline_accuracy, 3))
print("TF-IDF model accuracy:", round(model_accuracy, 3))
print("Macro F1:", round(macro_f1, 3))
print("Weighted F1:", round(weighted_f1, 3))

print("\nClassification report:")
print(report)

print("\nConfusion matrix (rows = actual, columns = predicted):")
print("Category order:", labels)
print(matrix)

# 7. Save the results
with open(results_path, "w", encoding="utf-8") as f:
    f.write("CampusIQ - Manual-Label Baseline Results\n")
    f.write("=" * 45 + "\n")
    f.write("Source: jntuh_policy_label_review.csv\n")
    f.write("Labels come from the manual review CSV; independent expert "
            "verification has not been established.\n")
    f.write(f"Examples used: {len(eligible)}\n")
    f.write(f"Training examples: {len(X_train)}\n")
    f.write(f"Test examples: {len(X_test)}\n")
    f.write(f"Evaluated categories: {labels}\n")
    f.write(f"Excluded categories: {excluded_categories.to_dict()}\n")
    f.write(f"Majority baseline accuracy: {baseline_accuracy:.3f}\n")
    f.write(f"TF-IDF model accuracy: {model_accuracy:.3f}\n")
    f.write(f"Macro F1: {macro_f1:.3f}\n")
    f.write(f"Weighted F1: {weighted_f1:.3f}\n\n")
    f.write("Classification report:\n")
    f.write(report)
    f.write("\nConfusion matrix (rows=actual, columns=predicted):\n")
    f.write(f"Category order: {labels}\n")
    f.write(str(matrix))
    f.write("\n\nCaution: the dataset is small. Results are preliminary "
            "and should not be treated as proof of reliable policy "
            "classification.\n")

print("\nResults saved to:", results_path)
print("\nReminder: these results are preliminary because the dataset is small.")
