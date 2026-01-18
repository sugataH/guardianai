"""
Docstring for guardianai.sidecar.sidecar_runtime
receives hooks
sends data to detectors (later)
builds alerts
publishes them to the event bus
optionally applies local actions
"""

from guardianai.sidecar.hooks import SidecarHooks
from guardianai.sidecar.local_actions import LocalActions
from guardianai.schemas.alert_schema import SecurityAlert, AlertEvidence
from guardianai.eventbus.bus import EventBus

#Agents imports from guardianai/agents/...
from guardianai.agents.prompt_inspector import PromptInspector
from guardianai.agents.decision_validator import OutputValidator
from guardianai.agents.comm_monitor import ToolUseMonitor
from guardianai.agents.data_integrity import MemoryWriteMonitor
from guardianai.agents.behavior_monitor import BehaviorMonitor
from guardianai.agents.resource_monitor import ResourceMonitor



class SidecarRuntime(SidecarHooks):
    """
    GuardianAI Sidecar runtime.
    This sits next to every operational AI agent.
    """

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

    def inspect_prompt(self, prompt: str):
        self.inspect_resource()
        self.inspect_behavior()
        risk = self.prompt_detector.analyze({"prompt": prompt})

        if risk > 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="prompt_injection",
                severity="high",
                confidence=risk,
                explanation="Prompt inspector flagged high risk",
                evidence=AlertEvidence(prompt_snippet=prompt),
            )

            # Always publish first
            self.bus.publish(self.sidecar_id, alert.model_dump())

            # Then block
            self.actions.block()

        return prompt

    def inspect_output(self, output: str):
        self.inspect_resource()
        self.inspect_behavior()
        risk = self.output_detector.analyze({"output": output})

        if risk > 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="unsafe_output",
                severity="high",
                confidence=risk,
                explanation="Sensitive data detected in output",
                evidence=AlertEvidence(output_snippet=output),
            )

            # Publish first
            self.bus.publish(self.sidecar_id, alert.model_dump())

            # Then block output
            self.actions.block()

        return output

    def inspect_tool(self, tool: str, args: dict):
        self.inspect_resource()
        self.inspect_behavior()
        risk = self.tool_detector.analyze({"tool": tool, "args": args})

        if risk > 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="tool_misuse",
                severity="high",
                confidence=risk,
                explanation="Suspicious or dangerous tool usage detected",
                evidence=AlertEvidence(tool_call={"tool": tool, "args": args}),
            )

            # Publish before blocking
            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.actions.block()

        return tool, args
    
    def inspect_memory_write(self, content: str):
        self.inspect_resource()
        self.inspect_behavior()
        risk = self.memory_detector.analyze({"memory": content})

        if risk > 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="memory_poisoning",
                severity="high",
                confidence=risk,
                explanation="Suspicious or unsafe memory write detected",
                evidence=AlertEvidence(memory_write=content),
            )

            # Publish before blocking
            self.bus.publish(self.sidecar_id, alert.model_dump())

            self.actions.block()

        return content
    
    def inspect_behavior(self):
        risk = self.behavior_detector.analyze(
            {"agent_id": self.agent_id}
        )

        if risk > 0.8:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="behavior_anomaly",
                severity="medium",
                confidence=risk,
                explanation="Abnormal behavior frequency detected",
                evidence=AlertEvidence(
                    behavior_metrics={
                        "event_count": float(self.behavior_detector.event_counts[self.agent_id])
                    }
                ),
            )

            self.bus.publish(self.sidecar_id, alert.model_dump())

    def inspect_resource(self):
        risk = self.resource_detector.analyze(
            {"agent_id": self.agent_id}
        )

        if risk > 0.8:
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
