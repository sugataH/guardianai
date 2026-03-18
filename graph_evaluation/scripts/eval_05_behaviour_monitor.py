"""
GuardianAI - BehaviourMonitor Evaluation Script
=================================================
Run from project root:
    python graph_evaluation/scripts/eval_05_behaviour_monitor.py

Output PNGs saved to:
    graph_evaluation/graph_png/behaviour_monitor/
"""

import os, sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
OUT_DIR = os.path.join(ROOT, "graph_evaluation", "graph_png", "behaviour_monitor")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, ROOT)
from guardianai.agents.behavior_monitor import BehaviorMonitor

STYLE = {
    "safe":   "#27AE60",
    "warn":   "#F39C12",
    "danger": "#C0392B",
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

# ── Figure 1: Score Evolution Under Attack Scenarios ─────────────────────────
print("Generating attack scenario evolution...")

scenarios = {
    "Normal diverse agent\n(varied actions)": {
        "actions": ["search", "calc", "translate", "read_file", "search",
                    "weather", "calc", "translate", "search", "summarize"],
        "color": STYLE["safe"],
    },
    "Repetition attack\n(same action repeating)": {
        "actions": ["inject_prompt"] * 15,
        "color": STYLE["danger"],
    },
    "Slow recon attack\n(sequential same type)": {
        "actions": ["read_file"] * 12,
        "color": STYLE["warn"],
    },
}

fig, ax = plt.subplots(figsize=(9, 5))
for scenario_name, scenario_data in scenarios.items():
    monitor = BehaviorMonitor()
    scores = []
    for action in scenario_data["actions"]:
        result = monitor.analyze({
            "agent_id": "test_agent",
            "action_type": action
        })
        scores.append(result["score"])
    ax.plot(range(1, len(scores)+1), scores,
            color=scenario_data["color"], lw=2.2,
            marker="o", markersize=5, label=scenario_name)

ax.axhline(0.85, color=STYLE["danger"], linestyle="--", lw=1.2,
           alpha=0.7, label="Block threshold (0.85)")
ax.axhline(0.55, color=STYLE["warn"], linestyle=":", lw=1.2,
           alpha=0.7, label="Warn threshold (0.55)")
ax.set_xlabel("Number of Actions"); ax.set_ylabel("Risk Score")
ax.set_ylim([-0.05, 1.1])
ax.set_title("BehaviourMonitor — Score Evolution Under Different Agent Patterns")
ax.legend(fontsize=9, framealpha=0.85)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bm_scenario_evolution.png"), dpi=150)
plt.close(fig)
print("Saved: bm_scenario_evolution.png")

# ── Figure 2: Signal Contribution Breakdown ───────────────────────────────────
print("Generating signal contribution breakdown...")

# Three patterns to compare
patterns = {
    "Normal\nDiverse": {
        "actions": ["search", "calc", "translate", "read_file",
                    "weather", "summarize", "calc", "search"],
        "color": STYLE["safe"],
    },
    "High\nRepetition": {
        "actions": ["inject"] * 10,
        "color": STYLE["danger"],
    },
    "Long\nSequence Run": {
        "actions": ["read_file"] * 20,
        "color": STYLE["warn"],
    },
}

signal_names = ["Rate\n(w=0.15)", "Repetition\n(w=0.35)",
                "Sequence\n(w=0.25)", "Entropy\n(w=0.25)"]
signal_keys  = ["rate_contribution", "repetition_contrib",
                "sequence_contrib",  "entropy_contrib"]

fig, axes = plt.subplots(1, 3, figsize=(11, 4.5), sharey=True)
for ax, (pname, pdata) in zip(axes, patterns.items()):
    monitor = BehaviorMonitor()
    result = None
    for action in pdata["actions"]:
        result = monitor.analyze({"agent_id": "test", "action_type": action})
    contribs = [result["signals"].get(k, 0) for k in signal_keys]
    bars = ax.bar(signal_names, contribs, color=pdata["color"],
                  alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, contribs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9.5)
    ax.set_title(pname, fontsize=11)
    ax.set_ylim([0, 1.0])
    if ax == axes[0]:
        ax.set_ylabel("Signal Contribution Score")
    ax.tick_params(axis="x", labelsize=8.5)

fig.suptitle("BehaviourMonitor — Signal Contributions by Pattern Type",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bm_signal_contributions.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: bm_signal_contributions.png")

# ── Figure 3: Entropy vs Score ─────────────────────────────────────────────────
print("Generating entropy relationship plot...")

# Sweep entropy by varying mix of action types
scores_by_diversity = []
diversities = []
for n_types in range(1, 9):
    monitor = BehaviorMonitor()
    actions = []
    for i in range(16):
        actions.append(f"action_{(i % n_types) + 1}")
    for a in actions:
        result = monitor.analyze({"agent_id": "test", "action_type": a})
    scores_by_diversity.append(result["score"])
    diversities.append(n_types)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(diversities, scores_by_diversity,
        color=STYLE["danger"], lw=2.2, marker="s", markersize=7)
ax.set_xlabel("Number of Distinct Action Types Used")
ax.set_ylabel("Final Risk Score")
ax.set_title("BehaviourMonitor — Action Diversity vs Risk Score\n"
             "(16 actions, same window, varying diversity)")
ax.set_xticks(diversities)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bm_diversity_vs_score.png"), dpi=150)
plt.close(fig)
print("Saved: bm_diversity_vs_score.png")

# ── Figure 4: Test Results Summary ───────────────────────────────────────────
test_classes = {
    "Normal\nBehavior\n(6)":   (6, 6),
    "Repetition\n(4)":         (4, 4),
    "Sequence\n(5)":           (5, 5),
    "Entropy\n(3)":            (3, 3),
    "Attack\nScenarios\n(4)":  (4, 4),
    "Agent\nIsolation\n(1)":   (1, 1),
    "Reset &\nProfile\n(4)":   (4, 4),
    "Edge\nCases\n(5)":        (5, 5),
    "TOTAL\n(32)":             (32, 32),
}
cats   = list(test_classes.keys())
passed = [v[0] for v in test_classes.values()]
total  = [v[1] for v in test_classes.values()]
bcolors = [STYLE["safe"]] * (len(cats)-1) + ["#2C3E50"]

fig, ax = plt.subplots(figsize=(11, 4))
bars = ax.bar(cats, [p/t*100 for p,t in zip(passed,total)],
              color=bcolors, edgecolor="white", width=0.65)
for bar, p, t in zip(bars, passed, total):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{p}/{t}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.set_ylim([0, 115])
ax.set_ylabel("Pass Rate (%)")
ax.set_title("BehaviourMonitor — Test Suite Results (32 tests, all passing)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.0f}%"))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bm_test_results.png"), dpi=150)
plt.close(fig)
print("Saved: bm_test_results.png")

print(f"\nAll graphs saved to:\n  {OUT_DIR}")
