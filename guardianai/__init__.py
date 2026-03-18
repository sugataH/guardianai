"""
GuardianAI — Runtime Security Framework for AI Agents
======================================================

GuardianAI provides a sidecar-based security layer that monitors, detects,
and responds to threats in multi-agent AI systems at runtime.

Architecture:
    Operational AI Agents
        ↓  (every prompt / output / tool call / memory write)
    SidecarRuntime  (one per agent)
        ↓  (six detectors, parallel evaluation)
    EventBus        (signed events, tamper-proof transport)
        ↓
    SupervisorAgent (correlation, policy, enforcement)
        ↓
    HITL Queue      (human review of quarantined agents)
        ↓
    AuditLogger     (Merkle-chained tamper-proof log)

Six Detection Layers:
    1. PromptInspector       — Hybrid rule + ML, detects prompt injection
    2. OutputValidator       — Hybrid rule + ML, detects data leakage
    3. ToolUseMonitor        — Rule-based, detects tool misuse
    4. MemoryWriteMonitor    — Hybrid statistical + rule + ML, detects poisoning
    5. BehaviorMonitor       — Statistical, detects behavioral anomalies
    6. ResourceMonitor       — Statistical, detects resource exhaustion
"""

__version__ = "1.0.0"
__author__  = "GuardianAI Research"
