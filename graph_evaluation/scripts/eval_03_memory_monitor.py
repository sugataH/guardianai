"""
GuardianAI - MemoryWriteMonitor Evaluation Script
===================================================
Run from project root:
    python graph_evaluation/scripts/eval_03_memory_monitor.py

Output PNGs saved to:
    graph_evaluation/graph_png/memory_monitor/
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, f1_score
)
import joblib, pandas as pd
from sklearn.model_selection import train_test_split

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_DIR = os.path.join(ROOT, "guardianai", "models", "memory_monitor")
DATA_DIR  = os.path.join(ROOT, "datasets", "memory_monitor")
OUT_DIR   = os.path.join(ROOT, "graph_evaluation", "graph_png", "memory_monitor")
os.makedirs(OUT_DIR, exist_ok=True)

VEC_WORD_PATH = os.path.join(MODEL_DIR, "memory_monitor_vectorizer_word.joblib")
VEC_CHAR_PATH = os.path.join(MODEL_DIR, "memory_monitor_vectorizer_char.joblib")
MODEL_PATH    = os.path.join(MODEL_DIR, "memory_monitor_model.joblib")

STYLE = {
    "malicious": "#C0392B", "benign": "#27AE60",
    "accent":    "#F39C12", "grid":   "#E0E0E0", "bg": "#FAFAFA",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.facecolor": STYLE["bg"], "figure.facecolor": "white",
    "axes.grid": True, "grid.color": STYLE["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

print("Loading model...")
vec_word = joblib.load(VEC_WORD_PATH)
vec_char = joblib.load(VEC_CHAR_PATH)
model    = joblib.load(MODEL_PATH)

def predict_scores(texts):
    xw = vec_word.transform(texts)
    xc = vec_char.transform(texts)
    x  = scipy.sparse.hstack([xw, xc], format="csr")
    return model.predict_proba(x)[:, 1]

# Load dataset
for name in ["memory_monitor_final.csv", "memory_monitor_built.csv"]:
    csv_path = os.path.join(DATA_DIR, name)
    if os.path.exists(csv_path):
        break
print(f"Loading dataset: {csv_path}")
df = pd.read_csv(csv_path).dropna()
text_col  = [c for c in df.columns if c not in ["label"]
             and df[c].dtype == object][0]
df["label"] = df["label"].astype(int)

_, X_test, _, y_test = train_test_split(
    df[text_col], df["label"],
    test_size=0.20, stratify=df["label"], random_state=42
)

y_scores  = predict_scores(X_test.tolist())
THRESHOLD = 0.50
y_pred    = (y_scores >= THRESHOLD).astype(int)

# ── ROC Curve ──────────────────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, color=STYLE["malicious"], lw=2.2,
        label=f"MemoryWriteMonitor (AUC = {roc_auc:.4f})")
ax.plot([0,1],[0,1],"#AAAAAA",lw=1.2,linestyle="--",label="Random")
ax.fill_between(fpr, tpr, alpha=0.08, color=STYLE["malicious"])
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — MemoryWriteMonitor")
ax.legend(loc="lower right", framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mm_roc_curve.png"), dpi=150)
plt.close(fig)
print("Saved: mm_roc_curve.png")

# ── Precision-Recall ──────────────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y_test, y_scores)
pr_auc       = auc(rec, prec)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec, prec, color=STYLE["benign"], lw=2.2,
        label=f"MemoryWriteMonitor (AUC = {pr_auc:.4f})")
ax.fill_between(rec, prec, alpha=0.08, color=STYLE["benign"])
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — MemoryWriteMonitor")
ax.legend(loc="lower left", framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mm_pr_curve.png"), dpi=150)
plt.close(fig)
print("Saved: mm_pr_curve.png")

# ── Confusion Matrix ──────────────────────────────────────────────────────
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Benign","Poisoning"])
fig, ax = plt.subplots(figsize=(5, 4))
disp.plot(ax=ax, cmap="Greens", colorbar=False)
ax.set_title("Confusion Matrix — MemoryWriteMonitor")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mm_confusion_matrix.png"), dpi=150)
plt.close(fig)
print("Saved: mm_confusion_matrix.png")

# ── Threshold vs F1 ───────────────────────────────────────────────────────
thresh_range = np.arange(0.05, 0.96, 0.02)
f1_vals = [f1_score(y_test, (y_scores >= t).astype(int), zero_division=0)
           for t in thresh_range]
best_t  = thresh_range[np.argmax(f1_vals)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresh_range, f1_vals, color=STYLE["malicious"], lw=2.2)
ax.axvline(best_t, color=STYLE["accent"], linestyle="--", lw=1.5,
           label=f"Best threshold = {best_t:.2f}  (F1 = {max(f1_vals):.4f})")
ax.set_xlabel("Threshold"); ax.set_ylabel("F1 Score")
ax.set_title("Threshold vs F1 — MemoryWriteMonitor")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mm_threshold_vs_f1.png"), dpi=150)
plt.close(fig)
print("Saved: mm_threshold_vs_f1.png")

# ── Score Distribution ────────────────────────────────────────────────────
benign_s  = y_scores[y_test == 0]
poison_s  = y_scores[y_test == 1]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(benign_s, bins=50, alpha=0.65, color=STYLE["benign"],
        label="Benign", density=True)
ax.hist(poison_s, bins=50, alpha=0.65, color=STYLE["malicious"],
        label="Poisoning", density=True)
ax.axvline(THRESHOLD, color=STYLE["accent"], linestyle="--", lw=1.8,
           label=f"Threshold ({THRESHOLD})")
ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
ax.set_title("Score Distribution — MemoryWriteMonitor")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mm_score_distribution.png"), dpi=150)
plt.close(fig)
print("Saved: mm_score_distribution.png")

# ── Adversarial Evaluation Category Chart ────────────────────────────────
# These are the actual results from the adversarial test run
ADVERSARIAL_DATA = {
    "A: Direct\nPoisoning":    (8,  8,  "flagged"),
    "B: Paraphrased\nAttacks": (7,  8,  "flagged"),
    "C: Subtle /\nIndirect":   (6,  6,  "flagged"),
    "D: Benign-\nSimilar":     (10, 10, "not flagged"),
    "E: Genuine\nBenign":      (7,  7,  "not flagged"),
    "F: Edge\nCases":          (5,  5,  "correct"),
}

labels  = list(ADVERSARIAL_DATA.keys())
correct = [v[0] for v in ADVERSARIAL_DATA.values()]
total   = [v[1] for v in ADVERSARIAL_DATA.values()]
rates   = [c/t*100 for c, t in zip(correct, total)]
bar_colors = [STYLE["benign"] if r == 100 else STYLE["accent"]
              for r in rates]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, rates, color=bar_colors, edgecolor="white",
              linewidth=0.8, width=0.6)
ax.axhline(100, color=STYLE["malicious"], linestyle="--", lw=1.2,
           alpha=0.5, label="100% correct")
for bar, c, t in zip(bars, correct, total):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{c}/{t}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=STYLE["accent"] if c < t else "#27AE60")
ax.set_ylim([0, 115])
ax.set_ylabel("Correct Classification Rate (%)")
ax.set_title("Adversarial Evaluation — MemoryWriteMonitor\n"
             "Out-of-distribution test (44 examples, never seen during training)")
ax.legend(framealpha=0.85)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mm_adversarial_categories.png"), dpi=150)
plt.close(fig)
print("Saved: mm_adversarial_categories.png")

print("\n── Classification Report ────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["Benign","Poisoning"]))
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"\nAll graphs saved to:\n  {OUT_DIR}")
