"""
guardianai.sidecar.sidecar_runtime

Sidecar Runtime Responsibilities:
- Intercepts agent operations
- Runs security detectors
- Builds SecurityAlert objects
- Publishes to EventBus
- Applies local mitigation when required
"""

from guardianai.sidecar.hooks import SidecarHooks
from guardianai.sidecar.local_actions import LocalActions
from guardianai.schemas.alert_schema import SecurityAlert, AlertEvidence
from guardianai.eventbus.bus import EventBus

# Detectors
from guardianai.agents.prompt_inspector import PromptInspector
from guardianai.agents.decision_validator import OutputValidator
from guardianai.agents.comm_monitor import ToolUseMonitor
from guardianai.agents.data_integrity import MemoryWriteMonitor
from guardianai.agents.behavior_monitor import BehaviorMonitor
from guardianai.agents.resource_monitor import ResourceMonitor

# Utils
from guardianai.utils.logger import get_logger
from guardianai.utils.metrics import Metrics


# ==============================
# GLOBAL THRESHOLDS
# ==============================
OPTIMAL_THRESHOLD = 0.45       # Scientifically optimized
HARD_BLOCK_THRESHOLD = 0.95    # Immediate deterministic block


class SidecarRuntime(SidecarHooks):
    """
    GuardianAI Sidecar Runtime.
    Deployed alongside every operational AI agent.
    """

    # =========================================================
    # INITIALIZATION
    # =========================================================
    def __init__(self, agent_id: str, sidecar_id: str, bus: EventBus):
        self.agent_id = agent_id
        self.sidecar_id = sidecar_id
        self.bus = bus
        self.actions = LocalActions()

        # Detectors
        self.prompt_detector = PromptInspector()
        self.output_detector = OutputValidator()
        self.tool_detector = ToolUseMonitor()
        self.memory_detector = MemoryWriteMonitor()
        self.behavior_detector = BehaviorMonitor()
        self.resource_detector = ResourceMonitor()

        # Utilities
        self.logger = get_logger(f"guardianai.sidecar.{self.sidecar_id}")
        self.metrics = Metrics()

    # =========================================================
    # PROMPT INSPECTION (Hybrid: Rule + ML)
    # =========================================================
    def inspect_prompt(self, prompt: str):
        self.metrics.inc("inspect.total")

        # Context monitoring
        self.inspect_resource()
        self.inspect_behavior()

        risk = self.prompt_detector.analyze({"prompt": prompt})

        # Severity mapping
        if risk < OPTIMAL_THRESHOLD:
            severity = "low"
        elif risk < 0.85:
            severity = "medium"
        else:
            severity = "high"

        alert = SecurityAlert(
            agent_id=self.agent_id,
            sidecar_id=self.sidecar_id,
            event_type="prompt_injection",
            severity=severity,
            confidence=risk,
            explanation="Hybrid rule + ML prompt risk score",
            evidence=AlertEvidence(prompt_snippet=prompt),
        )

        # Always notify Supervisor
        self.bus.publish(self.sidecar_id, alert.model_dump())

        # Immediate block only for extremely confident threats
        if risk >= HARD_BLOCK_THRESHOLD:
            self.logger.warning(
                f"block_prompt agent={self.agent_id} confidence={risk:.2f}"
            )
            self.metrics.inc("block.prompt")
            self.actions.block()

        return prompt

    # =========================================================
    # OUTPUT VALIDATION
    # =========================================================
    def inspect_output(self, output: str):
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        risk = self.output_detector.analyze({"output": output})

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="unsafe_output",
                severity="high",
                confidence=risk,
                explanation="Sensitive or policy-violating output detected",
                evidence=AlertEvidence(output_snippet=output),
            )

            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.logger.warning(
                f"block_output agent={self.agent_id} confidence={risk:.2f}"
            )
            self.metrics.inc("block.output")
            self.actions.block()

        return output

    # =========================================================
    # TOOL USAGE MONITOR
    # =========================================================
    def inspect_tool(self, tool: str, args: dict):
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        risk = self.tool_detector.analyze({"tool": tool, "args": args})

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="tool_misuse",
                severity="high",
                confidence=risk,
                explanation="Suspicious tool invocation detected",
                evidence=AlertEvidence(tool_call={"tool": tool, "args": args}),
            )

            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.logger.warning(
                f"block_tool agent={self.agent_id} confidence={risk:.2f}"
            )
            self.metrics.inc("block.tool")
            self.actions.block()

        return tool, args

    # =========================================================
    # MEMORY WRITE MONITOR
    # =========================================================
    def inspect_memory_write(self, content: str):
        self.metrics.inc("inspect.total")
        self.inspect_resource()
        self.inspect_behavior()

        risk = self.memory_detector.analyze({"memory": content})

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="memory_poisoning",
                severity="high",
                confidence=risk,
                explanation="Unsafe persistent memory modification detected",
                evidence=AlertEvidence(memory_write=content),
            )

            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.logger.warning(
                f"block_memory agent={self.agent_id} confidence={risk:.2f}"
            )
            self.metrics.inc("block.memory")
            self.actions.block()

        return content

    # =========================================================
    # BEHAVIOR MONITOR
    # =========================================================
    def inspect_behavior(self):
        risk = self.behavior_detector.analyze(
            {"agent_id": self.agent_id}
        )

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="behavior_anomaly",
                severity="medium",
                confidence=risk,
                explanation="Abnormal interaction frequency detected",
                evidence=AlertEvidence(
                    behavior_metrics={
                        "event_count": float(
                            self.behavior_detector.event_counts[self.agent_id]
                        )
                    }
                ),
            )

            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.logger.info(
                f"signal_behavior agent={self.agent_id} confidence={risk:.2f}"
            )

    # =========================================================
    # RESOURCE MONITOR
    # =========================================================
    def inspect_resource(self):
        risk = self.resource_detector.analyze(
            {"agent_id": self.agent_id}
        )

        if risk >= 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="resource_exhaustion",
                severity="high",
                confidence=risk,
                explanation="Excessive resource usage detected",
                evidence=AlertEvidence(
                    behavior_metrics={
                        "call_count": float(
                            self.resource_detector.call_counts[self.agent_id]
                        )
                    }
                ),
            )

            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.logger.info(
                f"signal_resource agent={self.agent_id} confidence={risk:.2f}"
            )
