"""
guardianai.sidecar.sidecar_runtime
====================================
Sidecar Runtime — intercepts all agent operations and runs
security detectors. One SidecarRuntime is deployed per agent.

Six detectors wired in order of complexity:

    Detector              File                  Method      Returns
    ─────────────────────────────────────────────────────────────────
    PromptInspector       prompt_inspector.py   Hybrid      float
    OutputValidator       decision_validator.py Hybrid      float
    ToolUseMonitor        comm_monitor.py       Rule-based  dict
    MemoryWriteMonitor    memory_write_monitor  Hybrid      float
    BehaviorMonitor       behavior_monitor.py   Statistical dict
    ResourceMonitor       resource_monitor.py   Statistical dict

NOTE on return types:
    PromptInspector, OutputValidator, MemoryWriteMonitor → float
    ToolUseMonitor, BehaviorMonitor, ResourceMonitor    → dict {"score": float, ...}
    All risk extraction uses _score() helper to handle both.
"""

from guardianai.sidecar.hooks import SidecarHooks
from guardianai.sidecar.local_actions import LocalActions
from guardianai.schemas.alert_schema import SecurityAlert, AlertEvidence
from guardianai.eventbus.bus import EventBus

# ── Detectors ────────────────────────────────────────────────────────────────
from guardianai.agents.prompt_inspector    import PromptInspector
from guardianai.agents.decision_validator  import OutputValidator
from guardianai.agents.comm_monitor        import ToolUseMonitor
from guardianai.agents.memory_write_monitor import MemoryWriteMonitor   # ← new
from guardianai.agents.behavior_monitor    import BehaviorMonitor
from guardianai.agents.resource_monitor    import ResourceMonitor

from guardianai.utils.logger  import get_logger
from guardianai.utils.metrics import Metrics


# ── Global thresholds ─────────────────────────────────────────────────────────
ALERT_THRESHOLD     = 0.45   # publish alert to Supervisor above this
HARD_BLOCK_THRESHOLD = 0.95  # immediate local block above this


def _score(result) -> float:
    """
    Normalise detector return value to a plain float.
    Handles both: float (PromptInspector, OutputValidator, MemoryWriteMonitor)
    and dict with 'score' key (ToolUseMonitor, BehaviorMonitor, ResourceMonitor).
    """
    if isinstance(result, dict):
        return float(result.get("score", 0.0))
    return float(result)


class SidecarRuntime(SidecarHooks):
    """
    GuardianAI Sidecar Runtime.
    Deployed alongside every operational AI agent.

    Intercept points:
        inspect_prompt(prompt)              → PromptInspector
        inspect_output(output)              → OutputValidator
        inspect_tool(tool, args)            → ToolUseMonitor
        inspect_memory_write(key, value)    → MemoryWriteMonitor
        inspect_behavior()                  → BehaviorMonitor  [called internally]
        inspect_resource()                  → ResourceMonitor  [called internally]
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self, agent_id: str, sidecar_id: str, bus: EventBus):
        self.agent_id   = agent_id
        self.sidecar_id = sidecar_id
        self.bus        = bus
        self.actions    = LocalActions()

        # ── Six detectors ────────────────────────────────────
        self.prompt_detector   = PromptInspector()
        self.output_detector   = OutputValidator()
        self.tool_detector     = ToolUseMonitor()
        self.memory_detector   = MemoryWriteMonitor()   # ← replaces data_integrity stub
        self.behavior_detector = BehaviorMonitor()
        self.resource_detector = ResourceMonitor()

        self.logger  = get_logger(f"guardianai.sidecar.{self.sidecar_id}")
        self.metrics = Metrics()

    # =========================================================
    # PROMPT INSPECTION
    # Hybrid: Rule-based + ML (PromptInspector)
    # Returns float — block at >= HARD_BLOCK_THRESHOLD
    # =========================================================

    def inspect_prompt(self, prompt: str) -> str:
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        risk = _score(self.prompt_detector.analyze({"prompt": prompt}))

        severity = (
            "high"   if risk >= 0.85 else
            "medium" if risk >= ALERT_THRESHOLD else
            "low"
        )

        alert = SecurityAlert(
            agent_id    = self.agent_id,
            sidecar_id  = self.sidecar_id,
            event_type  = "prompt_injection",
            severity    = severity,
            confidence  = risk,
            explanation = "Hybrid rule + ML prompt risk score",
            evidence    = AlertEvidence(prompt_snippet=prompt),
        )
        self.bus.publish(self.sidecar_id, alert.model_dump())

        if risk >= HARD_BLOCK_THRESHOLD:
            self.logger.warning(
                f"block_prompt agent={self.agent_id} confidence={risk:.3f}"
            )
            self.metrics.inc("block.prompt")
            self.actions.block()

        return prompt

    # =========================================================
    # OUTPUT VALIDATION
    # Hybrid: Rule-based + ML (OutputValidator)
    # Returns float — alert + block at >= 0.8
    # =========================================================

    def inspect_output(self, output: str) -> str:
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        risk = _score(self.output_detector.analyze({"output": output}))

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id    = self.agent_id,
                sidecar_id  = self.sidecar_id,
                event_type  = "unsafe_output",
                severity    = "high",
                confidence  = risk,
                explanation = "Sensitive or policy-violating output detected",
                evidence    = AlertEvidence(output_snippet=output),
            )
            self.bus.publish(self.sidecar_id, alert.model_dump())
            self.logger.warning(
                f"block_output agent={self.agent_id} confidence={risk:.3f}"
            )
            self.metrics.inc("block.output")
            self.actions.block()

        return output

    # =========================================================
    # TOOL USE INSPECTION
    # Rule-based allowlist/denylist + arg scanning (ToolUseMonitor)
    # Returns dict — alert + block at score >= 0.8
    # =========================================================

    def inspect_tool(self, tool: str, args: dict) -> tuple[str, dict]:
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        result = self.tool_detector.analyze({"tool": tool, "args": args})
        risk   = _score(result)

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id    = self.agent_id,
                sidecar_id  = self.sidecar_id,
                event_type  = "tool_misuse",
                severity    = "high",
                confidence  = risk,
                explanation = "Suspicious tool invocation detected",
                evidence    = AlertEvidence(
                    tool_call={"tool": tool, "args": args}
                ),
            )
            self.bus.publish(self.sidecar_id, alert.model_dump())
            self.logger.warning(
                f"block_tool agent={self.agent_id} "
                f"tool={tool} confidence={risk:.3f}"
            )
            self.metrics.inc("block.tool")
            self.actions.block()

        return tool, args

    # =========================================================
    # MEMORY WRITE INSPECTION
    # Hybrid: Statistical + Rule + ML (MemoryWriteMonitor)
    # Returns float — alert + block at score >= 0.8
    #
    # Signature change from old stub:
    #   OLD: inspect_memory_write(content: str)
    #        → passed {"memory": content}  to data_integrity stub
    #   NEW: inspect_memory_write(key: str, value: str)
    #        → passes {"agent_id", "key", "value"} to MemoryWriteMonitor
    #   Backward-compat: value-only call still works (key defaults to "")
    # =========================================================

    def inspect_memory_write(self,
                              key:   str = "",
                              value: str = "") -> str:
        """
        Intercept a memory write operation.

        Args:
            key   — the memory key being written to (e.g. "task_context")
            value — the content being written

        Old callers passing a single positional string still work:
            inspect_memory_write(content)  → treated as value, key=""
        """
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        risk = _score(
            self.memory_detector.analyze({
                "agent_id":   self.agent_id,
                "key":        key,
                "value":      value,
                "value_size": len(value.encode("utf-8")),
            })
        )

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id    = self.agent_id,
                sidecar_id  = self.sidecar_id,
                event_type  = "memory_poisoning",
                severity    = "high",
                confidence  = risk,
                explanation = "Unsafe persistent memory modification detected",
                evidence    = AlertEvidence(
                    memory_write=f"[key={key!r}] {value}"
                ),
            )
            self.bus.publish(self.sidecar_id, alert.model_dump())
            self.logger.warning(
                f"block_memory agent={self.agent_id} "
                f"key={key!r} confidence={risk:.3f}"
            )
            self.metrics.inc("block.memory")
            self.actions.block()

        return value

    # =========================================================
    # BEHAVIOR MONITORING
    # Statistical multi-signal (BehaviorMonitor)
    # Returns dict — called internally on every intercept
    # =========================================================

    def inspect_behavior(self) -> None:
        result = self.behavior_detector.analyze(
            {"agent_id": self.agent_id}
        )
        risk = _score(result)

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id    = self.agent_id,
                sidecar_id  = self.sidecar_id,
                event_type  = "behavior_anomaly",
                severity    = "medium",
                confidence  = risk,
                explanation = "Abnormal interaction frequency or pattern detected",
                evidence    = AlertEvidence(
                    behavior_metrics={
                        "event_count": float(
                            self.behavior_detector.event_counts[self.agent_id]
                        ),
                        "reasons": result.get("reasons", []),
                    }
                ),
            )
            self.bus.publish(self.sidecar_id, alert.model_dump())
            self.logger.info(
                f"signal_behavior agent={self.agent_id} confidence={risk:.3f} "
                f"reasons={result.get('reasons', [])}"
            )

    # =========================================================
    # RESOURCE MONITORING
    # Statistical sliding window (ResourceMonitor)
    # Returns dict — called internally on every intercept
    # =========================================================

    def inspect_resource(self) -> None:
        result = self.resource_detector.analyze(
            {"agent_id": self.agent_id}
        )
        risk = _score(result)

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id    = self.agent_id,
                sidecar_id  = self.sidecar_id,
                event_type  = "resource_exhaustion",
                severity    = "high",
                confidence  = risk,
                explanation = "Excessive resource usage detected",
                evidence    = AlertEvidence(
                    behavior_metrics={
                        "call_count": float(
                            result.get("signals", {})
                            .get("total_events_in_window", 0)
                        ),
                        "reasons": result.get("reasons", []),
                    }
                ),
            )
            self.bus.publish(self.sidecar_id, alert.model_dump())
            self.logger.info(
                f"signal_resource agent={self.agent_id} confidence={risk:.3f} "
                f"reasons={result.get('reasons', [])}"
            )
