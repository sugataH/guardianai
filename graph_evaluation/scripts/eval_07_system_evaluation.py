"""
GuardianAI - System-Level Evaluation Graphs Script
====================================================
Run from project root:
    python graph_evaluation/scripts/eval_07_system_evaluation.py

Generates graphs for:
  - Combined ML agent comparison (ROC overlay, F1 bar chart)
  - All 8 system scenarios
  - Test suite summary across all agents
  - Policy enforcement distribution
  - End-to-end latency comparison

Output PNGs saved to:
    graph_evaluation/graph_png/system_evaluation/
"""

import os, sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib, scipy.sparse, pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
OUT_DIR = os.path.join(ROOT, "graph_evaluation", "graph_png", "system_evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)

STYLE = {
    "pi":     "#C0392B",
    "ov":     "#2E86AB",
    "mm":     "#27AE60",
    "safe":   "#27AE60",
    "warn":   "#F39C12",
    "danger": "#C0392B",
    "purple": "#8E44AD",
    "navy":   "#1A252F",
    "grid":   "#E0E0E0",
    "bg":     "#FAFAFA",
    "text":   "#2C3E50",
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.facecolor": STYLE["bg"], "figure.facecolor": "white",
    "axes.grid": True, "grid.color": STYLE["grid"], "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load agent and dataset and compute ROC
# ─────────────────────────────────────────────────────────────────────────────
def load_pi():
    vec   = joblib.load(os.path.join(ROOT, "guardianai/models/prompt_inspector/prompt_vectorizer.joblib"))
    model = joblib.load(os.path.join(ROOT, "guardianai/models/prompt_inspector/prompt_model.joblib"))
    for name in ["guardianai_merged_cleaned.csv", "prompt_injection_clean.csv"]:
        p = os.path.join(ROOT, "datasets/prompt_injection", name)
        if os.path.exists(p):
            break
    df = pd.read_csv(p).dropna()
    tcol = "prompt" if "prompt" in df.columns else df.columns[0]
    df["label"] = df["label"].astype(int)
    _, X_test, _, y_test = train_test_split(
        df[tcol], df["label"], test_size=0.15,
        stratify=df["label"], random_state=42)
    X_v = vec.transform(X_test)
    return y_test.values, model.predict_proba(X_v)[:, 1]

def load_ov():
    vec   = joblib.load(os.path.join(ROOT, "guardianai/models/output_validator/output_validator_vectorizer.joblib"))
    model = joblib.load(os.path.join(ROOT, "guardianai/models/output_validator/output_validator_model.joblib"))
    for name in ["output_validator_final.csv", "output_validator_dataset.csv"]:
        p = os.path.join(ROOT, "datasets/output_validator", name)
        if os.path.exists(p):
            break
    df = pd.read_csv(p).dropna()
    tcol = [c for c in df.columns if c not in ["label","Label"]][0]
    lcol = "label" if "label" in df.columns else "Label"
    df[lcol] = df[lcol].astype(int)
    _, X_test, _, y_test = train_test_split(
        df[tcol], df[lcol], test_size=0.20,
        stratify=df[lcol], random_state=42)
    X_v = vec.transform(X_test)
    return y_test.values, model.predict_proba(X_v)[:, 1]

def load_mm():
    vw    = joblib.load(os.path.join(ROOT, "guardianai/models/memory_monitor/memory_monitor_vectorizer_word.joblib"))
    vc    = joblib.load(os.path.join(ROOT, "guardianai/models/memory_monitor/memory_monitor_vectorizer_char.joblib"))
    model = joblib.load(os.path.join(ROOT, "guardianai/models/memory_monitor/memory_monitor_model.joblib"))
    for name in ["memory_monitor_final.csv", "memory_monitor_built.csv"]:
        p = os.path.join(ROOT, "datasets/memory_monitor", name)
        if os.path.exists(p):
            break
    df = pd.read_csv(p).dropna()
    tcol = [c for c in df.columns if c not in ["label"] and df[c].dtype == object][0]
    df["label"] = df["label"].astype(int)
    _, X_test, _, y_test = train_test_split(
        df[tcol], df["label"], test_size=0.20,
        stratify=df["label"], random_state=42)
    xw = vw.transform(X_test.tolist())
    xc = vc.transform(X_test.tolist())
    x  = scipy.sparse.hstack([xw, xc], format="csr")
    return y_test.values, model.predict_proba(x)[:, 1]

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Combined ROC Overlay for 3 ML agents
# ─────────────────────────────────────────────────────────────────────────────
print("Loading models for combined ROC...")
pi_y, pi_s = load_pi()
ov_y, ov_s = load_ov()
mm_y, mm_s = load_mm()

fig, ax = plt.subplots(figsize=(7, 6))
for y, s, color, name in [
    (pi_y, pi_s, STYLE["pi"], "PromptInspector"),
    (ov_y, ov_s, STYLE["ov"], "OutputValidator"),
    (mm_y, mm_s, STYLE["mm"], "MemoryWriteMonitor"),
]:
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.2,
            label=f"{name}  (AUC = {roc_auc:.4f})")

ax.plot([0,1],[0,1],"#AAAAAA",lw=1.2,linestyle="--",label="Random classifier")
ax.fill_between([0,1],[0,1],alpha=0.04,color="#999999")
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Three ML-Based Agents")
ax.legend(loc="lower right", framealpha=0.9, fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_combined_roc.png"), dpi=150)
plt.close(fig)
print("Saved: sys_combined_roc.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: F1, Precision, Recall bar chart comparison
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.metrics import precision_score, recall_score, f1_score

agents = ["PromptInspector", "OutputValidator", "MemoryWriteMonitor"]
thresholds = [0.50, 0.45, 0.50]
ys = [pi_y, ov_y, mm_y]
ss = [pi_s, ov_s, mm_s]

metrics = {"Precision": [], "Recall": [], "F1 Score": []}
for y, s, t in zip(ys, ss, thresholds):
    pred = (s >= t).astype(int)
    metrics["Precision"].append(precision_score(y, pred))
    metrics["Recall"].append(recall_score(y, pred))
    metrics["F1 Score"].append(f1_score(y, pred))

x      = np.arange(len(agents))
width  = 0.25
mcolors= [STYLE["pi"], STYLE["ov"], STYLE["mm"]]

fig, ax = plt.subplots(figsize=(9, 5))
for i, (metric, vals) in enumerate(metrics.items()):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, vals, width, label=metric,
                  alpha=0.85, edgecolor="white",
                  color=[STYLE["pi"], STYLE["ov"], STYLE["mm"]][i])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x); ax.set_xticklabels(agents, fontsize=10)
ax.set_ylim([0.7, 1.05])
ax.set_ylabel("Score")
ax.set_title("ML Agent Performance Comparison — Precision, Recall, F1")
ax.legend(framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_ml_metrics_comparison.png"), dpi=150)
plt.close(fig)
print("Saved: sys_ml_metrics_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Test Suite Summary — all agents
# ─────────────────────────────────────────────────────────────────────────────
test_summary = {
    "PromptInspector\n(12 tests)":        (12, 12),
    "OutputValidator\n(43 tests)":        (43, 43),
    "ToolUseMonitor\n(40 tests)":         (40, 40),
    "MemoryWriteMonitor\n(adversarial)":  (41, 44),
    "BehaviourMonitor\n(32 tests)":       (32, 32),
    "ResourceMonitor\n(20 tests)":        (20, 20),
    "Supervisor\nOrchestration\n(62)":    (62, 62),
    "TOTAL\n(229 tests)":                 (250, 253),
}
cats   = list(test_summary.keys())
passed = [v[0] for v in test_summary.values()]
total  = [v[1] for v in test_summary.values()]
rates  = [p/t*100 for p, t in zip(passed, total)]
colors = [STYLE["safe"] if r >= 99 else STYLE["warn"] for r in rates[:-1]] + ["#2C3E50"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(cats, rates, color=colors, edgecolor="white", width=0.65)
for bar, p, t, r in zip(bars, passed, total, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{p}/{t}\n({r:.0f}%)", ha="center", va="bottom",
            fontsize=9, fontweight="bold")
ax.set_ylim([0, 118])
ax.set_ylabel("Pass Rate (%)")
ax.set_title("GuardianAI — Complete Test Suite Results Across All Agents")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_test_suite_summary.png"), dpi=150)
plt.close(fig)
print("Saved: sys_test_suite_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: System Scenario Results Summary
# ─────────────────────────────────────────────────────────────────────────────
scenario_data = {
    "S1: False\nPositive Rate":         100.0,   # 0% FPR = 100% correct pass
    "S2: Per-Detector\nDetection Rate": 100.0,
    "S3: Split Attack\nDetection":      100.0,
    "S4: Trust Decay\nDynamics":        100.0,
    "S5: HITL Recovery\nCorrectness":   100.0,
    "S6: Exponential\nDecay Accuracy":  99.3,    # 0.447 vs 0.450 = 99.3%
    "S7: Policy Tier\nDistribution":    100.0,
    "S8: Latency\nCompliance":          100.0,
}

cats    = list(scenario_data.keys())
values  = list(scenario_data.values())
bcolors = [STYLE["safe"] if v >= 99 else STYLE["warn"] for v in values]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(cats, values, color=bcolors, edgecolor="white", width=0.65)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim([94, 103])
ax.set_ylabel("Score / Pass Rate (%)")
ax.set_title("GuardianAI — System-Level Evaluation Scenario Results (8 Scenarios)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_scenario_results.png"), dpi=150)
plt.close(fig)
print("Saved: sys_scenario_results.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Latency Comparison
# ─────────────────────────────────────────────────────────────────────────────
from guardianai.agents.prompt_inspector    import PromptInspector
from guardianai.agents.decision_validator  import OutputValidator
from guardianai.agents.comm_monitor        import ToolUseMonitor
from guardianai.agents.memory_write_monitor import MemoryWriteMonitor
from guardianai.agents.behavior_monitor    import BehaviorMonitor
from guardianai.agents.resource_monitor    import ResourceMonitor

print("Measuring latency (20 calls each)...")
N = 20

agents_lat = {
    "PromptInspector":    (PromptInspector(),
                           {"prompt": "Write a Python function to sort a list"}),
    "OutputValidator":    (OutputValidator(),
                           {"output": "The task has been completed successfully."}),
    "ToolUseMonitor":     (ToolUseMonitor(),
                           {"tool": "search", "args": "query=weather"}),
    "MemoryWriteMonitor": (MemoryWriteMonitor(),
                           {"agent_id": "a1", "key": "user_pref",
                            "value": "User prefers concise responses.", "value_size": 40}),
    "BehaviourMonitor":   (BehaviorMonitor(),
                           {"agent_id": "a1", "action_type": "search"}),
    "ResourceMonitor":    (ResourceMonitor(),
                           {"agent_id": "a1"}),
}

lat_means = {}
lat_p95   = {}
for name, (agent, data) in agents_lat.items():
    times = []
    for _ in range(N):
        t0 = time.perf_counter()
        agent.analyze(data)
        times.append((time.perf_counter() - t0) * 1000)
    lat_means[name] = np.mean(times)
    lat_p95[name]   = np.percentile(times, 95)
    print(f"  {name}: mean={lat_means[name]:.3f}ms  p95={lat_p95[name]:.3f}ms")

names = list(lat_means.keys())
means = list(lat_means.values())
p95s  = list(lat_p95.values())
lat_colors = [STYLE["pi"], STYLE["ov"], STYLE["safe"],
              STYLE["mm"], STYLE["warn"], STYLE["purple"]]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(names))
bars = ax.bar(x - 0.2, means, 0.35, label="Mean latency",
              color=lat_colors, alpha=0.85, edgecolor="white")
ax.bar(x + 0.2, p95s, 0.35, label="P95 latency",
       color=lat_colors, alpha=0.45, edgecolor="white", hatch="//")
for bar, v in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
            f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("Monitor","Mon.") for n in names],
                   rotation=15, ha="right", fontsize=9.5)
ax.set_ylabel("Latency (ms)")
ax.set_title("GuardianAI — Per-Agent Inference Latency\n"
             f"(N={N} calls each; bars=mean, hatched=P95)")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_latency_comparison.png"), dpi=150)
plt.close(fig)
print("Saved: sys_latency_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Exponential Decay Verification
# ─────────────────────────────────────────────────────────────────────────────
import math
lambda_decay = math.log(2) / 30
t_range      = np.linspace(0, 90, 300)
theoretical  = 0.90 * np.exp(-lambda_decay * t_range)

# Measured points from system evaluation
measured_t      = [0,    30,    60]
measured_scores = [0.90, 0.447, 0.222]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(t_range, theoretical, color=STYLE["text"], lw=2.0,
        linestyle="--", label="Theoretical: 0.90 × e^(-λt)")
ax.scatter(measured_t, measured_scores, color=STYLE["danger"],
           s=100, zorder=4, label="Measured values")
for t, s in zip(measured_t, measured_scores):
    ax.annotate(f"  t={t}s: {s:.3f}", (t, s), fontsize=9.5,
                va="bottom", ha="left")
ax.axhline(0.35, color=STYLE["warn"], linestyle=":", lw=1.2, alpha=0.7,
           label="Warn threshold (0.35)")
ax.set_xlabel("Time Elapsed (seconds)")
ax.set_ylabel("Threat Score")
ax.set_ylim([-0.02, 1.0])
ax.set_title("Threat Correlation Engine — Exponential Decay Verification\n"
             f"(λ = ln2/30 = {lambda_decay:.4f} s⁻¹, half-life = 30s)")
ax.legend(framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_threat_decay.png"), dpi=150)
plt.close(fig)
print("Saved: sys_threat_decay.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Policy Tier Distribution
# ─────────────────────────────────────────────────────────────────────────────
tiers  = ["ALLOW\n(<0.35)",  "WARN\n(0.35-0.55)", "THROTTLE\n(0.55-0.75)",
          "BLOCK\n(0.75-0.90)", "QUARANTINE\n(≥0.90)"]
pcts   = [35, 20, 20, 15, 10]
colors = [STYLE["safe"], "#2ECC71", STYLE["warn"], STYLE["danger"], "#922B21"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
wedges, texts, autotexts = ax1.pie(
    pcts, labels=tiers, colors=colors, autopct='%1.0f%%',
    startangle=140, pctdistance=0.75,
    wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight("bold")
ax1.set_title("Policy Tier Distribution\n(threat score sweep 0→1)")

x = np.arange(len(tiers))
ax2.bar(x, pcts, color=colors, edgecolor="white", width=0.65)
for xi, v in zip(x, pcts):
    ax2.text(xi, v + 0.3, f"{v}%", ha="center", va="bottom",
             fontsize=10, fontweight="bold")
ax2.set_xticks(x); ax2.set_xticklabels(tiers, fontsize=8.5)
ax2.set_ylabel("Percentage of Score Range (%)")
ax2.set_title("Policy Tier Coverage")
ax2.set_ylim([0, 42])

fig.suptitle("GuardianAI — Policy Enforcement Tier Distribution",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_policy_distribution.png"), dpi=150)
plt.close(fig)
print("Saved: sys_policy_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Trust Score Dynamics
# ─────────────────────────────────────────────────────────────────────────────
# Simulate trust decay from transcript data
# Initial: 1.0, attacks degrade by 0.15 each (CRITICAL)
# Then recovery at +0.002/second

attack_times  = [0, 5, 10, 15, 20, 25]
trust_after_attacks = []
trust = 1.0
t = 0
timeline_t     = [0]
timeline_trust = [trust]
for at in attack_times:
    # recover from last event
    dt = at - t
    trust = min(1.0, trust + 0.002 * dt)
    timeline_t.append(at - 0.01)
    timeline_trust.append(trust)
    # apply attack
    trust = max(0.0, trust - 0.15)
    timeline_t.append(at)
    timeline_trust.append(trust)
    t = at

# HITL restore at t=35
restore_t     = 35
timeline_t.append(restore_t)
timeline_trust.append(0.70)

# Recovery after restore
for dt in range(1, 61):
    timeline_t.append(restore_t + dt)
    timeline_trust.append(min(1.0, 0.70 + 0.002 * dt))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(timeline_t, timeline_trust, color=STYLE["text"], lw=2.0)
ax.axvline(restore_t, color=STYLE["safe"], linestyle="--", lw=1.5,
           label=f"HITL restore at t={restore_t}s (trust reset to 0.70)")
ax.axhline(0.10, color=STYLE["danger"], linestyle=":", lw=1.2,
           alpha=0.7, label="Quarantine floor (0.10)")
ax.axhline(0.80, color=STYLE["warn"], linestyle=":", lw=1.2,
           alpha=0.7, label="Warn threshold (0.80)")
for at in attack_times:
    ax.axvline(at, color=STYLE["danger"], lw=0.8, alpha=0.35)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Trust Score")
ax.set_ylim([-0.05, 1.1])
ax.set_xlim([0, 95])
ax.set_title("Trust Store — Score Dynamics Under Attack and HITL Recovery\n"
             "(6 CRITICAL attacks, then HITL restore, then natural recovery)")
ax.legend(framealpha=0.85, fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "sys_trust_dynamics.png"), dpi=150)
plt.close(fig)
print("Saved: sys_trust_dynamics.png")

print(f"\n{'='*60}")
print(f"All system evaluation graphs saved to:\n  {OUT_DIR}")
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  graph_evaluation/graph_png/system_evaluation/{f}")
