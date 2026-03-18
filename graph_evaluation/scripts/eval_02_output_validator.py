"""
GuardianAI - OutputValidator Evaluation Script
================================================
Run from project root:
    python graph_evaluation/scripts/eval_02_output_validator.py

Output PNGs saved to:
    graph_evaluation/graph_png/output_validator/
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score
)
import joblib, pandas as pd
from sklearn.model_selection import train_test_split

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(ROOT, "guardianai", "models", "output_validator")
DATA_DIR  = os.path.join(ROOT, "datasets", "output_validator")
OUT_DIR   = os.path.join(ROOT, "graph_evaluation", "graph_png", "output_validator")
os.makedirs(OUT_DIR, exist_ok=True)

VEC_PATH   = os.path.join(MODEL_DIR, "output_validator_vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "output_validator_model.joblib")

STYLE = {
    "safe":     "#2E86AB",
    "unsafe":   "#C0392B",
    "accent":   "#F39C12",
    "grid":     "#E0E0E0",
    "bg":       "#FAFAFA",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.facecolor": STYLE["bg"], "figure.facecolor": "white",
    "axes.grid": True, "grid.color": STYLE["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

print("Loading model...")
vec   = joblib.load(VEC_PATH)
model = joblib.load(MODEL_PATH)

# Load dataset — try final first, then fallback
for name in ["output_validator_final.csv", "output_validator_dataset.csv"]:
    csv_path = os.path.join(DATA_DIR, name)
    if os.path.exists(csv_path):
        break
print(f"Loading dataset from {csv_path}...")
df = pd.read_csv(csv_path).dropna()

text_col  = [c for c in df.columns if c not in ["label", "Label"]][0]
label_col = "label" if "label" in df.columns else "Label"
df[label_col] = df[label_col].astype(int)

_, X_test, _, y_test = train_test_split(
    df[text_col], df[label_col],
    test_size=0.20, stratify=df[label_col], random_state=42
)

X_test_vec = vec.transform(X_test)
y_scores   = model.predict_proba(X_test_vec)[:, 1]
THRESHOLD  = 0.45
y_pred     = (y_scores >= THRESHOLD).astype(int)

# ── ROC Curve ─────────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=STYLE["unsafe"], lw=2.2,
        label=f"OutputValidator (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "#AAAAAA", lw=1.2, linestyle="--", label="Random")
ax.fill_between(fpr, tpr, alpha=0.08, color=STYLE["unsafe"])
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — OutputValidator")
ax.legend(loc="lower right", framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ov_roc_curve.png"), dpi=150)
plt.close(fig)
print("Saved: ov_roc_curve.png")

# ── Precision-Recall Curve ────────────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y_test, y_scores)
pr_auc       = auc(rec, prec)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=STYLE["safe"], lw=2.2,
        label=f"OutputValidator (AUC = {pr_auc:.4f})")
ax.fill_between(rec, prec, alpha=0.08, color=STYLE["safe"])
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — OutputValidator")
ax.legend(loc="lower left", framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ov_pr_curve.png"), dpi=150)
plt.close(fig)
print("Saved: ov_pr_curve.png")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Unsafe"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Reds", colorbar=False)
ax.set_title("Confusion Matrix — OutputValidator")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ov_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Saved: ov_confusion_matrix.png")

# ── Threshold vs F1 ───────────────────────────────────────────────────────────
thresh_range = np.arange(0.05, 0.96, 0.02)
f1_vals = [f1_score(y_test, (y_scores >= t).astype(int), zero_division=0)
           for t in thresh_range]
best_t  = thresh_range[np.argmax(f1_vals)]
best_f1 = max(f1_vals)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresh_range, f1_vals, color=STYLE["unsafe"], lw=2.2)
ax.axvline(best_t, color=STYLE["accent"], linestyle="--", lw=1.5,
           label=f"Best threshold = {best_t:.2f}  (F1 = {best_f1:.4f})")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
ax.set_title("Threshold vs F1 — OutputValidator")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ov_threshold_vs_f1.png"), dpi=150)
plt.close(fig)
print("Saved: ov_threshold_vs_f1.png")

# ── Score Distribution ────────────────────────────────────────────────────────
safe_s   = y_scores[y_test == 0]
unsafe_s = y_scores[y_test == 1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(safe_s,   bins=50, alpha=0.65, color=STYLE["safe"],   label="Safe",   density=True)
ax.hist(unsafe_s, bins=50, alpha=0.65, color=STYLE["unsafe"], label="Unsafe", density=True)
ax.axvline(THRESHOLD, color=STYLE["accent"], linestyle="--", lw=1.8,
           label=f"Threshold ({THRESHOLD})")
ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
ax.set_title("Score Distribution — OutputValidator")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "ov_score_distribution.png"), dpi=150)
plt.close(fig)
print("Saved: ov_score_distribution.png")

print("\n── Classification Report ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Safe", "Unsafe"]))
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"\nAll graphs saved to:\n  {OUT_DIR}")
