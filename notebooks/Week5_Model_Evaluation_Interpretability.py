"""Week 5: Evaluation, Validation and Interpretability — CampusIQ.
Input: data/processed/jntuh_policy_label_review.csv (columns: text, verified_category)
Run from repository root:
  python notebooks/Week5_Model_Evaluation_Interpretability.py
Outputs: reports/week5/
Labels are provisional until manually verified. Small-sample results are not deployment evidence.
"""
from pathlib import Path
import json, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVC

warnings.filterwarnings("default")
SEED, BOOTSTRAPS = 42, 1000
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/processed/jntuh_policy_label_review.csv"
OUT = ROOT / "reports/week5"
PLOTS, EXPL = OUT / "plots", OUT / "explanations"
for p in (OUT, PLOTS, EXPL): p.mkdir(parents=True, exist_ok=True)

def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

def term_count(text, terms):
    text = str(text).lower()
    return sum(len(re.findall(r"\\b" + re.escape(t) + r"\\b", text)) for t in terms)

def engineer(frame):
    frame = frame.copy()
    txt = frame["text"].fillna("").astype(str)
    frame["char_count"] = txt.str.len()
    frame["word_count"] = txt.apply(lambda s: len(re.findall(r"\\b\\w+\\b", s)))
    frame["sentence_count"] = txt.apply(lambda s: max(1, len(re.findall(r"[.!?]+", s))))
    frame["uppercase_ratio"] = txt.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
    frame["digit_ratio"] = txt.apply(lambda s: sum(c.isdigit() for c in s) / max(len(s), 1))
    groups = {
        "attendance_terms": ["attendance", "attend", "absent"],
        "exam_terms": ["examination", "examinations", "exam", "exams"],
        "credit_terms": ["credit", "credits", "grade", "grades", "marks"],
        "eligibility_terms": ["promotion", "eligibility", "detained"],
        "fee_terms": ["fee", "fees", "tuition"],
        "semester_terms": ["semester", "semesters"],
    }
    for name, terms in groups.items():
        frame[name] = txt.apply(lambda s: term_count(s, terms))
    return frame

NUM = ["char_count", "word_count", "sentence_count", "uppercase_ratio", "digit_ratio",
       "attendance_terms", "exam_terms", "credit_terms", "eligibility_terms", "fee_terms", "semester_terms"]

def models():
    baseline = Pipeline([
        ("features", ColumnTransformer([
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                      sublinear_tf=True), "text")
        ])),
        ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced",
                                          random_state=SEED))])
    prep = ColumnTransformer([
        ("word", TfidfVectorizer(max_features=5000, ngram_range=(1,2), sublinear_tf=True), "text"),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=3000,
                                 sublinear_tf=True), "text"),
        ("numeric", StandardScaler(), NUM)])
    advanced_lr = Pipeline([("features", prep),
        ("classifier", LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced", random_state=SEED))])
    advanced_svm = Pipeline([("features", prep),
        ("classifier", LinearSVC(C=0.1, class_weight="balanced", random_state=SEED))])
    return {
        "Word TF-IDF + Logistic Regression baseline": baseline,
        "Week 4 engineered Logistic Regression C=0.1": advanced_lr,
        "Week 4 engineered Linear SVM C=0.1": advanced_svm,
    }

def boot_ci(y, pred, reps=BOOTSTRAPS):
    rng = np.random.default_rng(SEED); y=np.asarray(y); pred=np.asarray(pred); vals=[]
    for _ in range(reps):
        ix = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[ix], pred[ix], average="macro", zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

def plot_cm(y, pred, labels, name):
    cm = confusion_matrix(y, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8,6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion matrix — " + name); fig.tight_layout()
    fig.savefig(PLOTS / ("confusion_" + slug(name) + ".png"), dpi=160); plt.close(fig)
    return cm

def plot_roc(model, X, y, classes, name):
    try:
        scores = model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)
        if scores.ndim == 1: return None
        ybin = label_binarize(y, classes=classes)
        fig, ax = plt.subplots(figsize=(7,6)); count=0
        for i, label in enumerate(classes):
            if len(np.unique(ybin[:,i])) < 2: continue
            fpr,tpr,_ = roc_curve(ybin[:,i], scores[:,i])
            ax.plot(fpr,tpr,label=f"{label} AUC={auc(fpr,tpr):.2f}"); count += 1
        if not count: plt.close(fig); return None
        ax.plot([0,1],[0,1],"--",label="Chance")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", title="One-vs-rest ROC — "+name)
        ax.legend(fontsize=7); fig.tight_layout()
        fig.savefig(PLOTS / ("roc_" + slug(name) + ".png"), dpi=160); plt.close(fig)
        return "See ROC plot (AUC reported per class)"
    except Exception as e:
        print("ROC skipped:", name, e); return None

def local_ablation(model, text, true_label, path):
    """Remove tokens one at a time and measure score changes; not causal."""
    def make_input(value):
        return engineer(pd.DataFrame({"text": [value]}))[["text"] + NUM]

    def score(value, predicted_label):
        X_one = make_input(value)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_one)[0]
            return float(probs[list(model.classes_).index(predicted_label)])
        vals = np.asarray(model.decision_function(X_one)).reshape(-1)
        if len(vals) == len(model.classes_):
            return float(vals[list(model.classes_).index(predicted_label)])
        return float(vals.max())

    predicted = model.predict(make_input(text))[0]
    base = score(text, predicted)
    tokens = re.findall(r"\\b[\\w'-]+\\b", str(text))
    impacts = []

    for i, token in enumerate(tokens):
        changed = " ".join(tokens[:i] + tokens[i+1:])
        impacts.append({
            "token": token,
            "score_drop_when_removed": base - score(changed, predicted)
        })

    impacts.sort(
        key=lambda item: abs(item["score_drop_when_removed"]),
        reverse=True
    )
    path.write_text(json.dumps({
        "text": text,
        "true_label": str(true_label),
        "predicted_label": str(predicted),
        "method": "Leave-one-token-out score ablation; descriptive, not causal.",
        "top_token_effects": impacts[:12]
    }, indent=2), encoding="utf-8")

def main():
    if not DATA.exists():
        raise SystemExit(f"Missing dataset: {DATA}\\nConfirm the CSV exists and run from the repository.")
    df = pd.read_csv(DATA)
    if not {"text","verified_category"}.issubset(df.columns):
        raise SystemExit("CSV must include text and verified_category columns.")
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df["verified_category"] = df["verified_category"].fillna("").astype(str).str.strip()
    df = df[(df.text!="") & (df.verified_category!="")].copy()
    counts = df.verified_category.value_counts()
    excluded = counts[counts < 3].to_dict()
    df = df[df.verified_category.isin(counts[counts >= 3].index)].copy()
    if df.verified_category.nunique()<2: raise SystemExit("Need at least two categories with 3+ examples.")
    df = engineer(df)
    X, y = df[["text"]+NUM], df["verified_category"]
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.25,random_state=SEED,stratify=y)
    classes = sorted(y.unique())
    min_count = int(ytr.value_counts().min())
    if min_count < 2: raise SystemExit("Too few training examples per category for cross-validation.")
    cv = StratifiedKFold(n_splits=min(3,min_count), shuffle=True, random_state=SEED)
    scoring = {"accuracy":"accuracy","macro_f1":"f1_macro","weighted_f1":"f1_weighted",
               "balanced_accuracy":"balanced_accuracy"}
    dummy=DummyClassifier(strategy="most_frequent").fit(Xtr.text,ytr)
    d_pred=dummy.predict(Xte.text)
    records=[{"model":"Majority-class baseline","test_accuracy":accuracy_score(yte,d_pred),
        "test_balanced_accuracy":balanced_accuracy_score(yte,d_pred),
        "test_macro_f1":f1_score(yte,d_pred,average="macro",zero_division=0),
        "test_weighted_f1":f1_score(yte,d_pred,average="weighted",zero_division=0),
        "bootstrap_macro_f1_ci_low":boot_ci(yte,d_pred)[0],"bootstrap_macro_f1_ci_high":boot_ci(yte,d_pred)[1],
        "cv_macro_f1_mean":np.nan,"cv_macro_f1_std":np.nan,"roc":"Not applicable"}]
    plot_cm(yte,d_pred,classes,"Majority-class baseline")
    for name, model in models().items():
        print("\\nEvaluating:", name)
        cvres = cross_validate(model,Xtr,ytr,cv=cv,scoring=scoring,n_jobs=1)
        fitted=clone(model).fit(Xtr,ytr); pred=fitted.predict(Xte)
        ci=boot_ci(yte,pred); macro=f1_score(yte,pred,average="macro",zero_division=0)
        cm=plot_cm(yte,pred,classes,name); roc=plot_roc(fitted,Xte,yte,classes,name)
        rep=classification_report(yte,pred,labels=classes,output_dict=True,zero_division=0)
        (OUT/("classification_"+slug(name)+".json")).write_text(json.dumps(rep,indent=2),encoding="utf-8")
        row={"model":name,"test_accuracy":accuracy_score(yte,pred),
            "test_balanced_accuracy":balanced_accuracy_score(yte,pred),"test_macro_f1":macro,
            "test_weighted_f1":f1_score(yte,pred,average="weighted",zero_division=0),
            "bootstrap_macro_f1_ci_low":ci[0],"bootstrap_macro_f1_ci_high":ci[1],
            "cv_macro_f1_mean":float(np.mean(cvres["test_macro_f1"])),
            "cv_macro_f1_std":float(np.std(cvres["test_macro_f1"],ddof=1)),
            "cv_accuracy_mean":float(np.mean(cvres["test_accuracy"])),
            "cv_balanced_accuracy_mean":float(np.mean(cvres["test_balanced_accuracy"])),
            "roc":roc or "Unavailable for this test split"}
        records.append(row)
        print(pd.Series(row).to_string())
        print(classification_report(yte,pred,labels=classes,zero_division=0))
        (OUT/("confusion_"+slug(name)+".json")).write_text(json.dumps({"classes":classes,"matrix":cm.tolist()},indent=2))
        # Permutation importance on raw input columns; noisy with a tiny test set.
        try:
            pi=permutation_importance(fitted,Xte,yte,scoring="f1_macro",n_repeats=10,random_state=SEED)
            imp=pd.DataFrame({"input_column":Xte.columns,"importance_mean":pi.importances_mean,
                              "importance_std":pi.importances_std}).sort_values("importance_mean",ascending=False)
            imp.to_csv(EXPL/("permutation_"+slug(name)+".csv"),index=False)
            show=imp.sort_values("importance_mean").tail(12)
            fig,ax=plt.subplots(figsize=(8,5)); ax.barh(show.input_column,show.importance_mean,xerr=show.importance_std)
            ax.set(title="Permutation importance — "+name,xlabel="Decrease in macro F1 after shuffling")
            fig.tight_layout(); fig.savefig(PLOTS/("importance_"+slug(name)+".png"),dpi=160); plt.close(fig)
        except Exception as e: print("Permutation importance skipped:",e)
        mis=np.flatnonzero(np.asarray(pred)!=np.asarray(yte))
        ix=int(mis[0]) if len(mis) else 0
        local_ablation(fitted,Xte.iloc[ix].text,yte.iloc[ix],EXPL/("local_"+slug(name)+".json"))
    results=pd.DataFrame(records); results.to_csv(OUT/"week5_metrics_comparison.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(results.model,results.test_macro_f1)
    ax.set(xlim=(0,1),xlabel="Test macro F1",title="Week 5 model comparison")
    fig.tight_layout(); fig.savefig(PLOTS/"model_comparison_macro_f1.png",dpi=160); plt.close(fig)
    summary={"dataset_examples":len(df),"train_examples":len(Xtr),"test_examples":len(Xte),
        "category_counts_before_filtering":counts.to_dict(),"excluded_categories_under_3_examples":excluded,
        "labels_independently_verified":False,"random_state":SEED,"bootstrap_resamples":BOOTSTRAPS,
        "results":records,"warning":"Small dataset and provisional labels; not deployment evidence."}
    (OUT/"week5_run_summary.json").write_text(json.dumps(summary,indent=2,default=str),encoding="utf-8")
    (OUT/"week5_evaluation_report.txt").write_text(
        "CampusIQ Week 5 Evaluation\\n==========================\\n"+
        f"Rows used: {len(df)}; train: {len(Xtr)}; test: {len(Xte)}\\n"+
        f"Excluded categories: {excluded}\\nLabels independently verified: NO\\n\\n"+
        results.to_string(index=False)+"\\n\\nSee plots/ and explanations/ for diagnostics.\\n"+
        "Local token ablation is descriptive, not causal. Do not claim deployment readiness.\\n",encoding="utf-8")
    print("\\nSaved actual metrics, plots, reports, and explanations to",OUT)

if __name__=="__main__": main()
