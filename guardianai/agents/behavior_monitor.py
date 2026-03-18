"""
guardianai.agents.behavior_monitor — BehaviorMonitor
=====================================================

Detection Method: Multi-signal statistical behavior profiling
                  (sliding window + action diversity + sequence analysis)
Event Type:       behavior_anomaly

Architecture:
    Agent performs any operation
        ↓
    SidecarRuntime.inspect_behavior()  [called on every sidecar interception]
        ↓
    BehaviorMonitor.analyze(data)
        ↓
    Risk score [0.0 – 1.0]
        ↓
    Alert → EventBus → Supervisor

Why NOT ML here:
    Behavioral anomaly detection is fundamentally an unsupervised problem at
    runtime — you don't have labeled "normal vs abnormal" behavior sequences
    per-agent at inference time. Statistical baselines (what is THIS agent's
    normal?) are both more accurate and more deployable than a fixed classifier
    trained on generic behavior. This is how SIEM and EDR systems work.

What the placeholder missed:
    The old code counted total lifetime events and compared to a hardcoded 5.
    This means any agent that legitimately processes 6 prompts would be flagged
    forever. It also had no concept of action types, sequence repetition,
    or time-windowed rate — just a global counter.

Design — Four Independent Signals:
    1. ACTION RATE       – events/sec in sliding window (same idea as ResourceMonitor
                           but operating on behavior events, not raw call counts)
    2. ACTION REPETITION – fraction of recent events that are the same action type
                           (detects prompt-retry loops, repeated injection attempts)
    3. SEQUENCE ANOMALY  – detects degenerate patterns: same action N times in a row
    4. ENTROPY DROP      – Shannon entropy of action distribution. A healthy agent
                           performs varied actions. An attacking agent does one thing
                           repeatedly → entropy collapses.

    Final score = weighted combination of all four signals.
    Weights favour repetition/entropy (semantic signals) over rate alone,
    because ResourceMonitor already owns the rate problem.

Sidecar compatibility:
    - analyze({"agent_id": ...})           ← minimum input (backward compatible)
    - analyze({"agent_id": ..., "action_type": ...})  ← richer input
    - event_counts[agent_id]               ← still exposed as public attribute
                                              so existing sidecar evidence code works
"""

import time
import math
from collections import defaultdict, deque
from guardianai.agents import Detector


# ===========================================================================
# THRESHOLDS
# ===========================================================================

# Sliding window for all signals
WINDOW_SECONDS: float = 15.0

# Action rate thresholds (events/sec) — complements ResourceMonitor
# BehaviorMonitor uses a slightly longer window and lower thresholds
# because it's looking for behavioral patterns, not raw DoS
RATE_MEDIUM: float   = 3.0   # events/sec → score contribution 0.25
RATE_HIGH: float     = 7.0   # events/sec → score contribution 0.55
RATE_CRITICAL: float = 15.0  # events/sec → score contribution 0.85

# Repetition — fraction of window events that share the same action_type
REPETITION_MEDIUM: float   = 0.60  # 60% same action  → contribution 0.30
REPETITION_HIGH: float     = 0.80  # 80% same action  → contribution 0.60
REPETITION_CRITICAL: float = 0.95  # 95% same action  → contribution 0.85
REPETITION_MIN_EVENTS: int = 5     # don't score repetition on tiny windows

# Sequence — consecutive identical actions in a row
SEQUENCE_MEDIUM: int   = 5   # N identical in a row → contribution 0.35
SEQUENCE_HIGH: int     = 10  # N identical in a row → contribution 0.65
SEQUENCE_CRITICAL: int = 20  # N identical in a row → contribution 0.90

# Entropy — Shannon entropy of action type distribution
# Max entropy for N action types = log2(N). We normalise to [0,1].
# A score of 0 = all events are one type (worst). 1 = perfectly uniform.
ENTROPY_LOW_THRESHOLD: float  = 0.25  # normalised entropy → contribution 0.30
ENTROPY_ZERO_THRESHOLD: float = 0.05  # near-zero entropy  → contribution 0.65
ENTROPY_MIN_EVENTS: int = 6           # need enough events to measure entropy

# Signal weights — must sum to 1.0
WEIGHT_RATE:       float = 0.15
WEIGHT_REPETITION: float = 0.35
WEIGHT_SEQUENCE:   float = 0.25
WEIGHT_ENTROPY:    float = 0.25

# Default action label when sidecar doesn't supply one
DEFAULT_ACTION: str = "generic"


# ===========================================================================
# BEHAVIOR MONITOR
# ===========================================================================

class BehaviorMonitor(Detector):
    """
    Detects abnormal agent behavior via statistical multi-signal profiling.

    Accepts:
        data = {
            "agent_id":    str   (required)
            "action_type": str   (optional — enriches all four signals)
        }

    Returns:
        dict with keys:
            score        (float)  – risk score 0.0–1.0
            blocked      (bool)   – True if score >= 0.85
            agent_id     (str)
            signals      (dict)   – per-signal breakdown
            reasons      (list)   – human-readable explanation

    Backward compatibility:
        event_counts[agent_id]  is still publicly accessible so existing
        sidecar evidence-collection code continues to work without changes.
    """

    def __init__(self):
        # Public: sidecar reads this directly for alert evidence
        self.event_counts: dict[str, int] = defaultdict(int)

        # Internal: sliding window of (timestamp, action_type) per agent
        self._windows: dict[str, deque] = defaultdict(deque)

        # Internal: consecutive identical action counter per agent
        self._last_action: dict[str, str] = {}
        self._consecutive: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def analyze(self, data: dict) -> dict:
        agent_id  = str(data.get("agent_id", "unknown"))
        action    = str(data.get("action_type", DEFAULT_ACTION)).lower().strip()
        now       = time.time()

        # --- Update public counter (sidecar compatibility) ---
        self.event_counts[agent_id] += 1

        # --- Update sliding window (evict stale) ---
        window = self._windows[agent_id]
        cutoff = now - WINDOW_SECONDS
        while window and window[0][0] < cutoff:
            window.popleft()
        window.append((now, action))

        # --- Update consecutive run counter ---
        if self._last_action.get(agent_id) == action:
            self._consecutive[agent_id] += 1
        else:
            self._consecutive[agent_id] = 1
            self._last_action[agent_id] = action

        total = len(window)
        reasons: list[str] = []

        # ------------------------------------------------------------------
        # SIGNAL 1: Action rate
        # ------------------------------------------------------------------
        rate = total / WINDOW_SECONDS
        rate_contrib, rate_reason = self._score_rate(rate)
        if rate_reason:
            reasons.append(rate_reason)

        # ------------------------------------------------------------------
        # SIGNAL 2: Action repetition
        # ------------------------------------------------------------------
        rep_contrib, rep_reason = self._score_repetition(window, total, action)
        if rep_reason:
            reasons.append(rep_reason)

        # ------------------------------------------------------------------
        # SIGNAL 3: Sequence (consecutive identical actions)
        # ------------------------------------------------------------------
        run = self._consecutive[agent_id]
        seq_contrib, seq_reason = self._score_sequence(run)
        if seq_reason:
            reasons.append(seq_reason)

        # ------------------------------------------------------------------
        # SIGNAL 4: Entropy drop
        # ------------------------------------------------------------------
        ent_contrib, ent_reason = self._score_entropy(window, total)
        if ent_reason:
            reasons.append(ent_reason)

        # ------------------------------------------------------------------
        # Weighted combination
        # ------------------------------------------------------------------
        final_score = (
            WEIGHT_RATE       * rate_contrib  +
            WEIGHT_REPETITION * rep_contrib   +
            WEIGHT_SEQUENCE   * seq_contrib   +
            WEIGHT_ENTROPY    * ent_contrib
        )
        final_score = round(min(final_score, 1.0), 4)

        if not reasons:
            reasons = ["normal behavior"]

        return {
            "score":    final_score,
            "blocked":  final_score >= 0.85,
            "agent_id": agent_id,
            "signals": {
                "action_rate_per_sec":   round(rate, 3),
                "rate_contribution":     round(rate_contrib, 4),
                "repetition_contrib":    round(rep_contrib, 4),
                "sequence_run":          run,
                "sequence_contrib":      round(seq_contrib, 4),
                "entropy_contrib":       round(ent_contrib, 4),
                "total_events_in_window": total,
                "dominant_action":       action,
            },
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # SIGNAL SCORERS
    # ------------------------------------------------------------------

    @staticmethod
    def _score_rate(rate: float) -> tuple[float, str]:
        if rate >= RATE_CRITICAL:
            return 0.85, f"critical action rate: {rate:.1f}/sec"
        elif rate >= RATE_HIGH:
            return 0.55, f"high action rate: {rate:.1f}/sec"
        elif rate >= RATE_MEDIUM:
            return 0.25, f"elevated action rate: {rate:.1f}/sec"
        return 0.05, ""

    @staticmethod
    def _score_repetition(window: deque, total: int, current_action: str) -> tuple[float, str]:
        if total < REPETITION_MIN_EVENTS:
            return 0.0, ""

        action_counts: dict[str, int] = defaultdict(int)
        for _, a in window:
            action_counts[a] += 1

        dominant      = max(action_counts, key=action_counts.get)
        dominant_frac = action_counts[dominant] / total

        if dominant_frac >= REPETITION_CRITICAL:
            return 0.85, (
                f"critical repetition: action '{dominant}' is "
                f"{dominant_frac:.0%} of window ({action_counts[dominant]}/{total})"
            )
        elif dominant_frac >= REPETITION_HIGH:
            return 0.60, (
                f"high repetition: action '{dominant}' is "
                f"{dominant_frac:.0%} of window ({action_counts[dominant]}/{total})"
            )
        elif dominant_frac >= REPETITION_MEDIUM:
            return 0.30, (
                f"elevated repetition: action '{dominant}' is "
                f"{dominant_frac:.0%} of window ({action_counts[dominant]}/{total})"
            )
        return 0.0, ""

    @staticmethod
    def _score_sequence(run: int) -> tuple[float, str]:
        if run >= SEQUENCE_CRITICAL:
            return 0.90, f"critical sequence: {run} identical actions in a row"
        elif run >= SEQUENCE_HIGH:
            return 0.65, f"high sequence run: {run} identical actions in a row"
        elif run >= SEQUENCE_MEDIUM:
            return 0.35, f"repeated sequence: {run} identical actions in a row"
        return 0.0, ""

    @staticmethod
    def _score_entropy(window: deque, total: int) -> tuple[float, str]:
        if total < ENTROPY_MIN_EVENTS:
            return 0.0, ""

        counts: dict[str, int] = defaultdict(int)
        for _, a in window:
            counts[a] += 1

        n_types = len(counts)
        if n_types == 1:
            # All events are the same type — minimum entropy
            return 0.65, f"zero entropy: all {total} events are action '{list(counts.keys())[0]}'"

        # Shannon entropy
        entropy = -sum(
            (c / total) * math.log2(c / total)
            for c in counts.values()
        )
        max_entropy = math.log2(n_types)
        normalised  = entropy / max_entropy if max_entropy > 0 else 1.0

        if normalised <= ENTROPY_ZERO_THRESHOLD:
            return 0.65, f"near-zero entropy: {normalised:.2f} normalised (action diversity collapsed)"
        elif normalised <= ENTROPY_LOW_THRESHOLD:
            return 0.30, f"low entropy: {normalised:.2f} normalised (low action diversity)"
        return 0.0, ""

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def reset_agent(self, agent_id: str):
        """Clear all state for an agent (call after quarantine/restart)."""
        self._windows.pop(agent_id, None)
        self._last_action.pop(agent_id, None)
        self._consecutive.pop(agent_id, None)
        self.event_counts[agent_id] = 0

    def get_agent_profile(self, agent_id: str) -> dict:
        """Return behavioral profile of an agent for audit/HITL."""
        window = self._windows.get(agent_id, deque())
        now    = time.time()
        cutoff = now - WINDOW_SECONDS
        active = [(t, a) for t, a in window if t >= cutoff]

        counts: dict[str, int] = defaultdict(int)
        for _, a in active:
            counts[a] += 1

        return {
            "agent_id":          agent_id,
            "events_in_window":  len(active),
            "action_types_seen": dict(counts),
            "consecutive_run":   self._consecutive.get(agent_id, 0),
            "last_action":       self._last_action.get(agent_id, None),
            "total_events_ever": self.event_counts[agent_id],
        }
