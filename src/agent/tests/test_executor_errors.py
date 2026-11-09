"""Executor handling of tool errors and invalid LLM-generated arguments."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.plan_execute.executor import (
    Executor,
    ToolCallError,
    _call_tool,
    _payload_error,
    _resolve_args_with_llm,
    _sanitize_args,
)
from agent.plan_execute.models import Plan, PlanStep, StepResult

_WO_PARAMS = [
    {"name": "site_id", "type": "string", "required": False},
    {"name": "asset_num", "type": "string", "required": False},
    {"name": "priority", "type": "integer", "required": False},
    {"name": "page_size", "type": "integer", "required": False},
]
_TOOLS = [
    {"name": "sites", "description": "List sites", "parameters": []},
    {"name": "list_workorders", "description": "List WOs", "parameters": _WO_PARAMS},
    {
        "name": "asset_detail",
        "description": "Asset detail",
        "parameters": [
            {"name": "site_name", "type": "string", "required": True},
            {"name": "asset_id", "type": "string", "required": True},
        ],
    },
]


def _step(n: int, tool: str) -> PlanStep:
    return PlanStep(
        step_number=n,
        task=f"Task {n}",
        server="iot",
        tool=tool,
        tool_args={},
        dependencies=[],
        expected_output="",
    )


async def _run_one(llm, tool: str, call_mock: AsyncMock) -> StepResult:
    executor = Executor(llm, server_paths={"iot": Path("/fake/server.py")})
    plan = Plan(steps=[_step(1, tool)], raw="")
    with (
        patch(
            "agent.plan_execute.executor._list_tools",
            new=AsyncMock(return_value=_TOOLS),
        ),
        patch("agent.plan_execute.executor._call_tool", new=call_mock),
    ):
        return (await executor.execute_plan(plan, "Q"))[0]


# ── _sanitize_args ────────────────────────────────────────────────────────────


def test_sanitize_drops_args_for_no_arg_tool():
    kept, dropped = _sanitize_args({"site_id": "site_1"}, [])
    assert kept == {} and dropped == {"site_id": "site_1"}


def test_sanitize_drops_none_and_placeholders_on_optional_params():
    args = {"site_id": "MAIN", "priority": "any?", "page_size": None, "asset_num": ""}
    kept, dropped = _sanitize_args(args, _WO_PARAMS)
    assert kept == {"site_id": "MAIN"}
    assert set(dropped) == {"priority", "page_size", "asset_num"}


def test_sanitize_keeps_real_values_and_required_params():
    params = [{"name": "asset_id", "type": "string", "required": True}]
    assert _sanitize_args({"asset_id": None}, params) == ({"asset_id": None}, {})
    kept, _ = _sanitize_args({"priority": 0, "page_size": 0}, _WO_PARAMS)
    assert kept == {"priority": 0, "page_size": 0}


# ── _payload_error ────────────────────────────────────────────────────────────


def test_payload_error_detection():
    assert _payload_error('{"error": "unknown asset_id X"}') == "unknown asset_id X"
    assert _payload_error('{"error": null, "sites": []}') is None
    assert _payload_error('{"sites": ["MAIN"]}') is None
    assert _payload_error("plain text") is None
    assert _payload_error("[1, 2]") is None


# ── execute_plan end to end (MCP mocked) ──────────────────────────────────────


@pytest.mark.anyio
async def test_error_payload_marks_step_failed(sequential_llm):
    body = json.dumps({"error": "unknown asset_id CWC04013 at site MAIN"})
    result = await _run_one(
        sequential_llm(['{"site_name": "MAIN", "asset_id": "CWC04013"}']),
        "asset_detail",
        AsyncMock(return_value=body),
    )
    assert result.success is False
    assert "unknown asset_id" in result.error
    assert result.response == body
    assert result.tool_args == {"site_name": "MAIN", "asset_id": "CWC04013"}


@pytest.mark.anyio
async def test_mcp_is_error_marks_step_failed(sequential_llm):
    call = AsyncMock(side_effect=ToolCallError("2 validation errors", "raw text"))
    result = await _run_one(sequential_llm(["{}"]), "sites", call)
    assert result.success is False
    assert "validation" in result.error
    assert result.response == "raw text"


@pytest.mark.anyio
async def test_invalid_args_are_not_sent(sequential_llm):
    call = AsyncMock(return_value='{"work_orders": []}')
    llm = sequential_llm(['{"site_id": "MAIN", "page_size": null, "priority": "any?"}'])
    result = await _run_one(llm, "list_workorders", call)
    assert call.await_args.args[2] == {"site_id": "MAIN"}
    assert result.tool_args == {"site_id": "MAIN"}
    assert result.success is True


@pytest.mark.anyio
async def test_hallucinated_args_on_no_arg_tool_are_dropped(sequential_llm):
    call = AsyncMock(return_value='{"sites": ["MAIN"]}')
    await _run_one(sequential_llm(['{"site_id": "site_1"}']), "sites", call)
    assert call.await_args.args[2] == {}


# ── _call_tool honours MCP isError ────────────────────────────────────────────


class _FakeSession:
    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def initialize(self):
        return None

    async def call_tool(self, name, args):
        return self._result


class _FakeStdio:
    async def __aenter__(self):
        return (MagicMock(), MagicMock())

    async def __aexit__(self, *exc):
        return False


@pytest.mark.anyio
@pytest.mark.parametrize("is_error", [True, False])
async def test_call_tool_raises_on_is_error(is_error):
    result = SimpleNamespace(
        content=[SimpleNamespace(text="Error executing tool")], isError=is_error
    )
    with (
        patch("mcp.client.stdio.stdio_client", return_value=_FakeStdio()),
        patch("mcp.ClientSession", return_value=_FakeSession(result)),
    ):
        if is_error:
            with pytest.raises(ToolCallError):
                await _call_tool(Path("/fake/server.py"), "t", {})
        else:
            assert await _call_tool(Path("/fake/server.py"), "t", {}) == (
                "Error executing tool"
            )


# ── failed steps are visible to later arg resolution ─────────────────────────


@pytest.mark.anyio
async def test_failed_step_shows_error_in_context():
    prompts: list[str] = []
    llm = MagicMock()
    llm.generate.side_effect = lambda prompt, **_: prompts.append(prompt) or "{}"
    context = {
        1: StepResult(step_number=1, task="t", server="iot", response="", error="boom")
    }
    await _resolve_args_with_llm("Q", "task", "tool", "", context, llm)
    assert "Step 1: ERROR: boom" in prompts[0]
