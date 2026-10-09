# Week 3 - Baseline Machine Learning Model
# CampusIQ Academic Policy RAG

from pathlib import Path
import re
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the processed policy chunks
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "processed" / "jntuh_policy_chunks.csv"

df = pd.read_csv(data_path)
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()

print("Dataset shape:", df.shape)

# 2. Create transparent, rule-based demonstration labels
def assign_category(text):
    text = text.lower()

    rules = [
        ("attendance", r"\battendance\b|\battend\b"),
        ("examinations", r"\bexamination\b|\bexaminations\b|\bexam\b|\bexams\b"),
        ("grading_credits", r"\bgrades?\b|\bcredits?\b|\bmarks\b|\bgrading\b"),
        ("promotion_eligibility", r"\bpromotion\b|\beligibility\b|\bdetained\b"),
        ("academic_calendar", r"\bsemester\b|\bacademic year\b|\bworking days\b"),
        ("fees", r"\bfees?\b|\bfee\b|\btuition\b"),
    ]

    for category, pattern in rules:
        if re.search(pattern, text):
            return category

    return "general"

df["category"] = df["text"].apply(assign_category)

print("\nRule-assigned category counts:")
print(df["category"].value_counts())

# 3. Check whether a stratified split is feasible
class_counts = df["category"].value_counts()
eligible = df[df["category"].isin(class_counts[class_counts >= 2].index)].copy()

if len(eligible) < 10 or eligible["category"].nunique() < 2:
    raise SystemExit(
        "\nNot enough labeled examples for a meaningful train/test split. "
        "Inspect category counts and collect manually verified labels before training."
    )

# Keep only categories with at least 2 examples for this prototype
X = eligible["text"]
y = eligible["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4. Majority-class benchmark
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
baseline_predictions = dummy.predict(X_test)

# 5. TF-IDF + Logistic Regression model
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
    ),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 6. Evaluate
print("\nTest examples:", len(y_test))
print("Majority-class baseline accuracy:", round(accuracy_score(y_test, baseline_predictions), 3))
print("TF-IDF + Logistic Regression accuracy:", round(accuracy_score(y_test, predictions), 3))
print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

# 7. Save results
results_path = project_dir / "data" / "processed" / "week3_baseline_results.txt"
with open(results_path, "w", encoding="utf-8") as f:
    f.write("Week 3 Baseline Model Results\n")
    f.write("Labels are rule-assigned demonstrations, not verified ground truth.\n")
    f.write(f"Original chunks: {len(df)}\n")
    f.write(f"Eligible chunks used: {len(eligible)}\n")
    f.write(f"Train examples: {len(X_train)}\n")
    f.write(f"Test examples: {len(X_test)}\n")
    f.write(f"Majority baseline accuracy: {accuracy_score(y_test, baseline_predictions):.3f}\n")
    f.write(f"TF-IDF Logistic Regression accuracy: {accuracy_score(y_test, predictions):.3f}\n")
    f.write("\n")
    f.write(classification_report(y_test, predictions, zero_division=0))

print("\nResults saved to:", results_path)
print("Reminder: these scores evaluate heuristic labels, not verified policy categories.")
