from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, atomic_write_text, load_json


_CODE_BLOCK = re.compile(r"```(?:python|py)\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.IGNORECASE | re.DOTALL)


def extract_python(text: str) -> str:
    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        raise ValueError("No fenced Python block found in agent response")
    return blocks[-1].strip() + "\n"


def extract_json_object(text: str) -> dict[str, Any]:
    blocks = _JSON_BLOCK.findall(text)
    if not blocks:
        raise ValueError("No fenced JSON object found in agent response")
    value = json.loads(blocks[-1])
    if not isinstance(value, dict):
        raise ValueError("Agent JSON result must be an object")
    return value


def _message_text(message: Any) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    texts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            texts.append(str(block.get("text", "")))
    return "\n".join(texts)


def _load_config(config_path: Path, role: str, system_prompt: str) -> dict[str, Any]:
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    config = copy.deepcopy(config)
    wrapped = config.get("agent_config")
    root = wrapped if isinstance(wrapped, dict) else config
    agent_args = root.setdefault("agent_args", {})
    agent_args["agent_name"] = f"pm_{role}_agent"
    agent_args["sys_prompt"] = system_prompt
    # PM synthesis is panel-code generation; the CAD/BRep MCP tools in the legacy
    # config are unrelated and make the role impossible to invoke independently.
    root["tool_args"] = None
    return config


async def run_agent_role(
    role: str,
    workspace: Path,
    round_number: int,
    config_path: Path,
    prompts_path: Path | None = None,
) -> Path:
    if role not in {"domain", "coding", "verifier"}:
        raise ValueError(f"Unsupported role {role!r}")
    prompts_path = prompts_path or Path(__file__).with_name("prompts.json")
    prompts = load_json(prompts_path)
    role_prompt = prompts[role]
    role_dir = workspace / "rounds" / f"round-{round_number:03d}" / "agents" / role
    request_path = role_dir / "request.json"
    request = load_json(request_path)
    request_json = json.dumps(request, ensure_ascii=False, indent=2)
    user_prompt = role_prompt["user"].format(request_json=request_json)

    try:
        from agentscope.message import Msg, TextBlock
        from agentscope.pipeline import stream_printing_messages
        from agent.create_react_agent import load_agent_from_config
        from agent.trajectory_vis import parse_message, process_conversation
    except ImportError as exc:
        raise RuntimeError("AgentScope runtime is unavailable in the selected Python environment") from exc

    config = _load_config(config_path, role, role_prompt["system"])
    agent = await load_agent_from_config(config)
    agent.set_console_output_enabled(False)
    initial = Msg("user", [TextBlock(type="text", text=user_prompt)], "user")
    history = [initial]
    async for message, last in stream_printing_messages(
        agents=[agent], coroutine_task=agent(initial)
    ):
        if last:
            history.append(message)

    raw_messages: list[dict[str, Any]] = []
    for message in history:
        raw_messages.extend(parse_message(message))
    atomic_write_json(role_dir / "conversation.json", raw_messages)
    process_conversation(history, "conversation", str(role_dir))

    assistant_text = "\n".join(
        _message_text(message) for message in history if message.role == "assistant"
    )
    atomic_write_text(role_dir / "response.md", assistant_text)
    if role == "coding":
        result_path = role_dir / "candidate.py"
        atomic_write_text(result_path, extract_python(assistant_text))
    else:
        result_path = role_dir / "result.json"
        atomic_write_json(result_path, extract_json_object(assistant_text))
    atomic_write_json(role_dir / "status.json", {"status": "success", "result": result_path.name})
    return result_path
