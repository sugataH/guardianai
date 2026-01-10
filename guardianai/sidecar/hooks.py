"""
Docstring for guardianai.sidecar.hooks
This file defines hook points that an AI agent must call:
before a prompt
after a response
before tool use
before memory write
"""

from typing import Any, Dict


class SidecarHooks:
    """
    Hook points that operational AI agents must call.
    The sidecar inspects everything here.
    """

    def on_prompt(self, agent_id: str, prompt: str) -> Dict[str, Any]:
        return {"agent_id": agent_id, "prompt": prompt}

    def on_output(self, agent_id: str, output: str) -> Dict[str, Any]:
        return {"agent_id": agent_id, "output": output}

    def on_tool_call(self, agent_id: str, tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {"agent_id": agent_id, "tool": tool, "args": args}

    def on_memory_write(self, agent_id: str, content: str) -> Dict[str, Any]:
        return {"agent_id": agent_id, "memory": content}
