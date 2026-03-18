"""
guardianai.sidecar.hooks
========================
Hook points that every operational AI agent must call.
The SidecarRuntime overrides these to inject security monitoring.

Usage pattern (operational agent):
    class MyAgent:
        def __init__(self, sidecar: SidecarRuntime):
            self.sidecar = sidecar

        def process_prompt(self, prompt: str) -> str:
            self.sidecar.inspect_prompt(prompt)       # ← hook
            # ... agent logic ...
            self.sidecar.inspect_output(output)       # ← hook
            return output

        def use_tool(self, tool: str, args: dict):
            self.sidecar.inspect_tool(tool, args)     # ← hook

        def write_memory(self, key: str, value: str):
            self.sidecar.inspect_memory_write(key, value)  # ← hook
"""

from typing import Any, Dict


class SidecarHooks:
    """
    Hook points for sidecar interception.
    Overridden by SidecarRuntime — stubs here allow agents to run
    without a sidecar attached (testing / development mode).
    """

    def inspect_prompt(self, prompt: str) -> str:
        return prompt

    def inspect_output(self, output: str) -> str:
        return output

    def inspect_tool(self, tool: str, args: Dict[str, Any]) -> tuple:
        return tool, args

    def inspect_memory_write(self, key: str = "", value: str = "") -> str:
        return value

    def inspect_behavior(self) -> None:
        pass

    def inspect_resource(self) -> None:
        pass
