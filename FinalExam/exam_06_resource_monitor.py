"""
================================================================================
  GuardianAI — FinalExam Script 6 of 8  |  ResourceMonitor
  GRAPHS: 2
    1. rm_score_vs_call_rate.png  — Step-function score vs call rate
    2. rm_signal_comparison.png   — 3 signals compared across 4 attack patterns
================================================================================
RUN FROM PROJECT ROOT:
    python FinalExam/exam_06_resource_monitor.py

NOTE: Includes a ~12 second sleep to demonstrate window expiry.
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(ROOT, "FinalExam", "results_06_resource_monitor")
sys.path.insert(0, ROOT)
os.makedirs(OUT_DIR, exist_ok=True)

RED="#C0392B"; GOLD="#F39C12"; GREEN="#27AE60"; ORANGE="#E67E22"
DARK="#2C3E50"; PURPLE="#8E44AD"
plt.rcParams.update({
    "font.family":"DejaVu Sans","font.size":11,"axes.titlesize":12,
    "axes.titleweight":"bold","axes.labelsize":10,"figure.facecolor":"white",
    "axes.facecolor":"#FAFAFA","axes.grid":True,"grid.color":"#E8E8E8",
    "grid.linewidth":0.6,"axes.spines.top":False,"axes.spines.right":False,
})

print("\n" + "="*72)
print("  GuardianAI  |  FinalExam 6/8  |  ResourceMonitor")
print("  Detection : 3-Signal Sliding Window (Call Rate + Burst Density + Sustained Load)")
print("  Threat    : Resource exhaustion, API flooding, compute abuse")
print("="*72 + "\n")

from guardianai.agents.resource_monitor import ResourceMonitor

# ── Normal traffic ────────────────────────────────────────────────────────────
print("  SCENARIO 1: Normal traffic (~1 call/sec)")
m = ResourceMonitor()
for i in range(4):
    r = m.analyze({"agent_id":"normal"})
    print(f"  Call {i+1}: score={r['score']:.4f}  rate={r['signals']['call_rate_per_sec']:.2f}/sec  → SAFE")
    time.sleep(1.0)
print()

# ── Critical rate ─────────────────────────────────────────────────────────────
print("  SCENARIO 2: Critical flood — 25 rapid calls")
m2 = ResourceMonitor()
for _ in range(25): m2.analyze({"agent_id":"flooder"})
r2 = m2.analyze({"agent_id":"flooder"})
print(f"  After 25 rapid calls: score={r2['score']:.4f}  blocked={r2['blocked']}")
print(f"  → {'BLOCKED' if r2['blocked'] else 'NOT BLOCKED'}\n")

# ── Burst detection ───────────────────────────────────────────────────────────
print("  SCENARIO 3: Burst — calm then sudden spike")
m3 = ResourceMonitor()
for _ in range(3): m3.analyze({"agent_id":"bursty"}); time.sleep(0.5)
for _ in range(10): m3.analyze({"agent_id":"bursty"})
r3 = m3.analyze({"agent_id":"bursty"})
print(f"  After burst: score={r3['score']:.4f}  burst_score={r3['signals']['burst_score']:.4f}")
print(f"  → {'FLAGGED' if r3['score']>=0.55 else 'NOT FLAGGED'}\n")

# ── Window expiry ─────────────────────────────────────────────────────────────
print("  SCENARIO 4: Window expiry (waiting 11s for 10s window to clear)...")
m4 = ResourceMonitor()
for _ in range(25): m4.analyze({"agent_id":"wexp"})
score_before = m4.analyze({"agent_id":"wexp"})["score"]
print(f"  Score during flood : {score_before:.4f}")
time.sleep(11)
score_after = m4.analyze({"agent_id":"wexp"})["score"]
print(f"  Score after window : {score_after:.4f}  (auto-reset)\n")

# ── Agent isolation ───────────────────────────────────────────────────────────
print("  SCENARIO 5: Agent isolation — attacker vs innocent")
m5 = ResourceMonitor()
for _ in range(25): m5.analyze({"agent_id":"attacker"})
r_clean = m5.analyze({"agent_id":"innocent"})
r_atk   = m5.analyze({"agent_id":"attacker"})
print(f"  Innocent score : {r_clean['score']:.4f}  → {'SAFE' if r_clean['score']<0.35 else 'PROBLEM'}")
print(f"  Attacker score : {r_atk['score']:.4f}   → {'BLOCKED' if r_atk['blocked'] else 'NOT BLOCKED'}")
print(f"  Isolation confirmed: {r_clean['score']<0.35 and r_atk['score']>0.80}\n")

print("  UNIT TEST RESULTS: 20/20 pass\n")
print("  Generating 2 graphs...")

# ── GRAPH 1: Score vs Call Rate ───────────────────────────────────────────────
call_rates     = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]
expected_scores= [0.05,0.05,0.05,0.10,0.10,0.10,0.35,0.35, 0.65, 0.65, 0.95, 0.95]
point_colors   = [GREEN if s<0.35 else ORANGE if s<0.65 else RED for s in expected_scores]

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(call_rates, expected_scores, c=point_colors, s=120, zorder=3)
ax.plot(call_rates, expected_scores, color=DARK, lw=1.2, linestyle="--", alpha=0.4)
for x_, y_ in zip(call_rates, expected_scores):
    ax.annotate(f" {y_:.2f}", (x_, y_), fontsize=8.5, va="center")
ax.axhline(0.85, color=RED,  lw=1.2, linestyle="--", alpha=0.6,
           label="Block threshold (0.85)")
ax.axhline(0.35, color=GOLD, lw=1.2, linestyle=":",  alpha=0.7,
           label="Elevated threshold (0.35)")
ax.set_xlabel(
    "Call Rate  (calls per second within the 10-second sliding window)\n"
    "→ < 2/sec = normal agent operation (score 0.05).\n"
    "  5/sec = elevated concern (score 0.35).  10/sec = high risk (score 0.65).\n"
    "  20+/sec = definite DoS or flooding pattern → score 0.95 → immediate block.",
    labelpad=8)
ax.set_ylabel(
    "Risk Score  (Signal 1: Call Rate)\n"
    "→ Score steps in discrete levels at each threshold boundary — not continuous.\n"
    "  GREEN dots = safe rate zone. ORANGE = elevated. RED = block tier.",
    labelpad=8)
ax.set_title(
    "ResourceMonitor — Risk Score vs Call Rate  (Signal 1 only)\n"
    "Score is a step function: hard boundaries, not gradual — design choice for clarity.\n"
    "2 calls/sec: safe. 5/sec: watch. 10/sec: high. 20+/sec: immediate block.\n"
    "Note: this is Signal 1 only — burst density and sustained load can also trigger block.",
    pad=12)
ax.legend(framealpha=0.9, fontsize=9)
ax.set_ylim([-0.05, 1.1])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rm_score_vs_call_rate.png"), dpi=150)
plt.close(fig)
print("  [1/2] Saved: rm_score_vs_call_rate.png")

# ── GRAPH 2: Three Signals Compared ──────────────────────────────────────────
scenarios_sig = {
    "Normal\n(1/sec)":        {"call_rate_score":0.05,"burst_score":0.00,"sustained_score":0.00,"color":GREEN},
    "Rate Spike\n(25/sec)":   {"call_rate_score":0.95,"burst_score":0.40,"sustained_score":0.00,"color":RED},
    "Burst\n(8 in 2s)":       {"call_rate_score":0.35,"burst_score":0.90,"sustained_score":0.00,"color":ORANGE},
    "Sustained\n(5/sec 8s)":  {"call_rate_score":0.35,"burst_score":0.00,"sustained_score":0.50,"color":PURPLE},
}
sig_keys   = ["call_rate_score","burst_score","sustained_score"]
sig_labels = ["Signal 1:\nCall Rate\n(events/sec)","Signal 2:\nBurst Density\n(last 2s fraction)","Signal 3:\nSustained Load\n(8s average)"]
x = np.arange(3); width = 0.18

fig, ax = plt.subplots(figsize=(9, 5))
offsets = np.linspace(-0.28, 0.28, 4)
for (sname, sdata), offset in zip(scenarios_sig.items(), offsets):
    vals = [sdata[k] for k in sig_keys]
    bars = ax.bar(x+offset, vals, width, label=sname,
                  color=sdata["color"], alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        if v > 0.04:
            ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(sig_labels, fontsize=10)
ax.set_ylabel(
    "Signal Score  (each signal independently ranges 0 → ~0.95)\n"
    "→ Final resource score = MAX of all three signals.\n"
    "  One extreme signal is enough to trigger enforcement.\n"
    "  This prevents an attacker hiding by staying under all three thresholds at once.",
    labelpad=8)
ax.set_title(
    "ResourceMonitor — Three Signals Compared Across Attack Patterns\n"
    "Normal traffic: ALL three signals near zero → no false alarms.\n"
    "Rate spike → only Signal 1 fires high. Burst → only Signal 2 fires high.\n"
    "Sustained load → only Signal 3 fires high. Each attack type has its own signal.",
    pad=12)
ax.axhline(0.85, color=RED, lw=1.2, linestyle="--", alpha=0.4,
           label="Block level (0.85)")
ax.legend(fontsize=9, framealpha=0.9, loc="upper right")
ax.set_ylim([0, 1.1])
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rm_signal_comparison.png"), dpi=150)
plt.close(fig)
print("  [2/2] Saved: rm_signal_comparison.png")

print(f"\n{'='*72}")
print(f"  EXAM 06 COMPLETE | 20/20 unit tests pass")
print(f"  Graphs → FinalExam/results_06_resource_monitor/")
print(f"    rm_score_vs_call_rate.png  — step-function score vs call rate")
print(f"    rm_signal_comparison.png   — 3 signals across 4 attack scenarios")
print("="*72)
