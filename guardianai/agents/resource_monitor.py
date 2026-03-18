"""
guardianai.agents.resource_monitor — ResourceMonitor
=====================================================

Detection Method: Multi-signal sliding window + adaptive rate thresholding
Event Type:       resource_exhaustion

Architecture:
    Agent performs any operation
        ↓
    SidecarRuntime.inspect_resource()
        ↓
    ResourceMonitor.analyze()
        ↓
    Risk score [0.0 – 1.0]
        ↓
    Alert → EventBus → Supervisor

Why NOT ML here:
    Resource abuse is a quantitative, real-time signal problem.
    ML models need training data, introduce latency, and are overkill
    for what is fundamentally a rate-limiting problem.
    Statistical thresholds on a sliding window are precise, instant,
    explainable, and impossible to "fool" through semantic variation.

Design:
    Each agent gets its own independent sliding window.
    The window stores timestamps of recent events (deque of floats).
    On each call, stale timestamps (older than WINDOW_SECONDS) are
    evicted, then the current event is appended.

    Three signals are measured independently and contribute to score:

    1. CALL RATE       – raw events per second within the window
    2. BURST DENSITY   – ratio of events in the last SHORT_BURST_SECONDS
                         vs the full window (detects sudden spikes)
    3. SUSTAINED LOAD  – whether the agent has been consistently active
                         for longer than SUSTAINED_LOAD_SECONDS

    Final score = max(call_rate_score, burst_score, sustained_score)
    This avoids masking: a single extreme signal is enough to flag.

Risk Score Bands:
    0.0 – 0.29  →  LOW      (normal operation)
    0.3 – 0.59  →  MEDIUM   (elevated activity, watch)
    0.6 – 0.84  →  HIGH     (significant abuse pattern)
    0.85 – 1.0  →  CRITICAL (definitive resource exhaustion attack)
"""

import time
from collections import defaultdict, deque
from guardianai.agents import Detector


# ===========================================================================
# THRESHOLDS — tunable per deployment
# ===========================================================================

# Sliding window size in seconds — history kept per agent
WINDOW_SECONDS: float = 10.0

# Short burst window — used for burst density detection
SHORT_BURST_SECONDS: float = 2.0

# Sustained load window — agent must be active for this long to trigger
SUSTAINED_LOAD_SECONDS: float = 8.0

# Call rate thresholds (calls per second within WINDOW_SECONDS)
RATE_LOW: float    = 2.0   # below this → score 0.1  (normal)
RATE_MEDIUM: float = 5.0   # below this → score 0.35 (elevated)
RATE_HIGH: float   = 10.0  # below this → score 0.65 (high)
RATE_CRITICAL: float = 20.0  # above this → score 0.95 (definitive abuse)

# Burst density threshold — fraction of total window events in burst window
# e.g. 0.7 means 70% of all events happened in the last SHORT_BURST_SECONDS
BURST_DENSITY_THRESHOLD: float = 0.70
BURST_MIN_EVENTS: int = 5  # don't trigger burst on very few events

# Sustained load threshold — calls per second over SUSTAINED_LOAD_SECONDS
SUSTAINED_RATE_THRESHOLD: float = 4.0


# ===========================================================================
# RESOURCE MONITOR
# ===========================================================================

class ResourceMonitor(Detector):
    """
    Detects resource exhaustion attacks via multi-signal sliding window analysis.

    Tracks per-agent call history using a deque of timestamps.
    Computes three independent scores and returns the maximum.

    Returns:
        dict with keys:
            score            (float)  – risk score 0.0–1.0
            blocked          (bool)   – True if score >= 0.85
            agent_id         (str)
            signals          (dict)   – per-signal breakdown for audit
            reasons          (list)   – human-readable explanation
    """

    def __init__(self):
        # Per-agent deque of event timestamps (evicted when stale)
        self._windows: dict[str, deque] = defaultdict(deque)

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def analyze(self, data: dict) -> dict:
        agent_id = str(data.get("agent_id", "unknown"))
        now = time.time()

        window = self._windows[agent_id]

        # --- Evict stale events ---
        cutoff = now - WINDOW_SECONDS
        while window and window[0] < cutoff:
            window.popleft()

        # --- Record current event ---
        window.append(now)

        total_events = len(window)
        elapsed = WINDOW_SECONDS  # normalise always against full window

        # ------------------------------------------------------------------
        # SIGNAL 1: Call rate (events per second in full window)
        # ------------------------------------------------------------------
        call_rate = total_events / elapsed
        rate_score, rate_reason = self._score_call_rate(call_rate)

        # ------------------------------------------------------------------
        # SIGNAL 2: Burst density (short window vs full window)
        # ------------------------------------------------------------------
        burst_score, burst_reason = self._score_burst_density(window, now, total_events)

        # ------------------------------------------------------------------
        # SIGNAL 3: Sustained load (continuous activity over time)
        # ------------------------------------------------------------------
        sustained_score, sustained_reason = self._score_sustained_load(window, now, total_events)

        # ------------------------------------------------------------------
        # Final score = maximum of all signals (worst-case wins)
        # ------------------------------------------------------------------
        final_score = max(rate_score, burst_score, sustained_score)
        final_score = round(min(final_score, 1.0), 4)

        reasons = [r for r in [rate_reason, burst_reason, sustained_reason] if r]
        if not reasons:
            reasons = ["normal operation"]

        signals = {
            "call_rate_per_sec": round(call_rate, 3),
            "call_rate_score": round(rate_score, 4),
            "burst_score": round(burst_score, 4),
            "sustained_score": round(sustained_score, 4),
            "total_events_in_window": total_events,
        }

        return {
            "score": final_score,
            "blocked": final_score >= 0.85,
            "agent_id": agent_id,
            "signals": signals,
            "reasons": reasons,
        }

    # ------------------------------------------------------------------
    # SIGNAL SCORERS
    # ------------------------------------------------------------------

    @staticmethod
    def _score_call_rate(rate: float) -> tuple[float, str]:
        """Score based on raw call rate (events/sec)."""
        if rate >= RATE_CRITICAL:
            return 0.95, f"critical call rate: {rate:.1f} calls/sec (threshold {RATE_CRITICAL})"
        elif rate >= RATE_HIGH:
            return 0.65, f"high call rate: {rate:.1f} calls/sec (threshold {RATE_HIGH})"
        elif rate >= RATE_MEDIUM:
            return 0.35, f"elevated call rate: {rate:.1f} calls/sec (threshold {RATE_MEDIUM})"
        elif rate >= RATE_LOW:
            return 0.10, ""
        else:
            return 0.05, ""

    @staticmethod
    def _score_burst_density(window: deque, now: float, total: int) -> tuple[float, str]:
        """Score based on concentration of events in the short burst window."""
        if total < BURST_MIN_EVENTS:
            return 0.0, ""

        burst_cutoff = now - SHORT_BURST_SECONDS
        burst_count = sum(1 for t in window if t >= burst_cutoff)
        density = burst_count / total if total > 0 else 0.0

        if density >= BURST_DENSITY_THRESHOLD and burst_count >= BURST_MIN_EVENTS:
            score = min(0.4 + density * 0.5, 0.95)
            return score, (
                f"burst detected: {burst_count}/{total} events in last "
                f"{SHORT_BURST_SECONDS}s ({density:.0%} density)"
            )
        return 0.0, ""

    @staticmethod
    def _score_sustained_load(window: deque, now: float, total: int) -> tuple[float, str]:
        """Score based on sustained high-rate activity over a longer period."""
        if total < 2:
            return 0.0, ""

        sustained_cutoff = now - SUSTAINED_LOAD_SECONDS
        sustained_events = sum(1 for t in window if t >= sustained_cutoff)
        sustained_rate = sustained_events / SUSTAINED_LOAD_SECONDS

        if sustained_rate >= SUSTAINED_RATE_THRESHOLD:
            score = min(0.3 + (sustained_rate / SUSTAINED_RATE_THRESHOLD) * 0.4, 0.90)
            return score, (
                f"sustained load: {sustained_rate:.1f} calls/sec over "
                f"{SUSTAINED_LOAD_SECONDS}s (threshold {SUSTAINED_RATE_THRESHOLD})"
            )
        return 0.0, ""

    # ------------------------------------------------------------------
    # UTILITY
    # ------------------------------------------------------------------

    def reset_agent(self, agent_id: str):
        """Clear history for an agent (call after quarantine/restart)."""
        if agent_id in self._windows:
            self._windows[agent_id].clear()

    def get_agent_stats(self, agent_id: str) -> dict:
        """Return current window stats for an agent (for audit/HITL)."""
        window = self._windows.get(agent_id, deque())
        now = time.time()
        cutoff = now - WINDOW_SECONDS
        active = [t for t in window if t >= cutoff]
        return {
            "agent_id": agent_id,
            "events_in_window": len(active),
            "window_seconds": WINDOW_SECONDS,
            "current_rate": round(len(active) / WINDOW_SECONDS, 3),
        }
