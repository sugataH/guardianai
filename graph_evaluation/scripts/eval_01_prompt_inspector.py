"""
GuardianAI - PromptInspector Evaluation Script
================================================
Run from project root:
    python graph_evaluation/scripts/eval_01_prompt_inspector.py

Output PNGs saved to:
    graph_evaluation/graph_png/prompt_inspector/
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR  = os.path.join(ROOT, "guardianai", "models", "prompt_inspector")
DATA_DIR   = os.path.join(ROOT, "datasets", "prompt_injection")
OUT_DIR    = os.path.join(ROOT, "graph_evaluation", "graph_png", "prompt_inspector")
os.makedirs(OUT_DIR, exist_ok=True)

VEC_PATH   = os.path.join(MODEL_DIR, "prompt_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "prompt_model.joblib")

STYLE = {
    "malicious":  "#C0392B",
    "benign":     "#2E86AB",
    "accent":     "#F39C12",
    "grid":       "#E0E0E0",
    "bg":         "#FAFAFA",
    "text":       "#2C3E50",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.facecolor": STYLE["bg"],
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.color": STYLE["grid"],
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Load model ───────────────────────────────────────────────────────────────
print("Loading model...")
vec   = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)

# ── Re-create test set ───────────────────────────────────────────────────────
import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = os.path.join(DATA_DIR, "guardianai_merged_cleaned.csv")
if not os.path.exists(csv_path):
    # fallback to other possible name
    csv_path = os.path.join(DATA_DIR, "prompt_injection_clean.csv")

print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path)
# Determine text column
text_col  = "prompt" if "prompt" in df.columns else df.columns[0]
label_col = "label"

df = df[[text_col, label_col]].dropna()
df[label_col] = df[label_col].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    df[text_col], df[label_col],
    test_size=0.15, stratify=df[label_col], random_state=42
)

X_test_vec = vec.transform(X_test)
y_scores   = model.predict_proba(X_test_vec)[:, 1]
y_pred     = (y_scores >= 0.50).astype(int)

# ── Figure 1: ROC Curve ──────────────────────────────────────────────────────
print("Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=STYLE["malicious"], lw=2.2,
        label=f"PromptInspector (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="#AAAAAA", lw=1.2, linestyle="--", label="Random classifier")
ax.fill_between(fpr, tpr, alpha=0.08, color=STYLE["malicious"])
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — PromptInspector")
ax.legend(loc="lower right", framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pi_roc_curve.png"), dpi=150)
plt.close(fig)

# ── Figure 2: Precision-Recall Curve ─────────────────────────────────────────
print("Generating Precision-Recall curve...")
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(recall, precision, color=STYLE["benign"], lw=2.2,
        label=f"PromptInspector (AUC = {pr_auc:.4f})")
ax.fill_between(recall, precision, alpha=0.08, color=STYLE["benign"])
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — PromptInspector")
ax.legend(loc="lower left", framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pi_pr_curve.png"), dpi=150)
plt.close(fig)

# ── Figure 3: Confusion Matrix ────────────────────────────────────────────────
print("Generating Confusion Matrix...")
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Benign", "Injection"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title("Confusion Matrix — PromptInspector")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pi_confusion_matrix.png"), dpi=150)
plt.close(fig)

# ── Figure 4: Threshold vs F1 ─────────────────────────────────────────────────
print("Generating Threshold vs F1 curve...")
from sklearn.metrics import f1_score
thresholds_sweep = np.arange(0.05, 0.96, 0.02)
f1_scores = [f1_score(y_test, (y_scores >= t).astype(int), zero_division=0)
             for t in thresholds_sweep]
best_t  = thresholds_sweep[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds_sweep, f1_scores, color=STYLE["malicious"], lw=2.2)
ax.axvline(best_t, color=STYLE["accent"], linestyle="--", lw=1.5,
           label=f"Best threshold = {best_t:.2f}  (F1 = {best_f1:.4f})")
ax.set_xlabel("Classification Threshold"); ax.set_ylabel("F1 Score")
ax.set_title("Threshold vs F1 — PromptInspector")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pi_threshold_vs_f1.png"), dpi=150)
plt.close(fig)

# ── Figure 5: Score Distribution ─────────────────────────────────────────────
print("Generating Score Distribution...")
benign_scores    = y_scores[y_test == 0]
injection_scores = y_scores[y_test == 1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(benign_scores,    bins=50, alpha=0.65, color=STYLE["benign"],
        label="Benign",    density=True)
ax.hist(injection_scores, bins=50, alpha=0.65, color=STYLE["malicious"],
        label="Injection", density=True)
ax.axvline(0.50, color=STYLE["accent"], linestyle="--", lw=1.8,
           label="Decision threshold (0.50)")
ax.set_xlabel("Predicted Probability Score"); ax.set_ylabel("Density")
ax.set_title("Score Distribution — PromptInspector")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "pi_score_distribution.png"), dpi=150)
plt.close(fig)

# ── Summary ───────────────────────────────────────────────────────────────────
report = classification_report(y_test, y_pred,
                                target_names=["Benign", "Injection"])
print("\n── Classification Report ────────────────────────────────")
print(report)
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"\nAll graphs saved to:\n  {OUT_DIR}")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  graph_evaluation/graph_png/prompt_inspector/{f}")
