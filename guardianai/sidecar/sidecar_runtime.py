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
from guardianai.agents.prompt_inspector import PromptInspector


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
        self.prompt_detector = PromptInspector()

    def inspect_prompt(self, prompt: str):
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
        return self.actions.allow(output)

    def inspect_tool(self, tool: str, args: dict):
        if "curl" in tool or "wget" in tool:
            alert = SecurityAlert(
                agent_id=self.agent_id,
                sidecar_id=self.sidecar_id,
                event_type="tool_misuse",
                severity="high",
                confidence=0.9,
                explanation="Suspicious external command",
                evidence=AlertEvidence(tool_call={"tool": tool, "args": args}),
            )
            self.bus.publish(self.sidecar_id, alert.model_dump())
            self.actions.block()
        return tool, args
