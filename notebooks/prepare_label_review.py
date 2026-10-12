from pathlib import Path
import re
import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
input_path = project_dir / "data" / "processed" / "jntuh_policy_chunks.csv"
output_path = project_dir / "data" / "processed" / "jntuh_policy_label_review.csv"

df = pd.read_csv(input_path)
df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()

def suggest_category(text):
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

df["suggested_category"] = df["text"].apply(suggest_category)

review = df.copy()
review["verified_category"] = ""

review.to_csv(output_path, index=False, encoding="utf-8")

print("Label-review file created successfully.")
print("Rows available for review:", len(review))
print("\nSuggested category counts:")
print(review["suggested_category"].value_counts())
print("\nSaved to:", output_path)
print("\nIMPORTANT: Suggested categories are not verified labels.")
