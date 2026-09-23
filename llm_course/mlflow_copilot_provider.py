"""GitHub Copilot CLI as an MLflow Assistant provider.

MLflow 3.16's Assistant ships providers for Claude Code, Codex CLI, Ollama and
OpenAI-compatible servers, in a hard-coded list. This module adds a provider that drives
the official GitHub Copilot CLI the same way the Claude Code provider drives `claude -p`:
one non-interactive `copilot -p ... --output-format json` process per turn, resumed with
`--resume` on the next turn, with MLflow's skills available to it.

`install()` registers the provider by wrapping MLflow's provider factories in the current
process. `tools/mlflow_server.py` calls it before starting the server in-process, so the
Assistant's provider menu lists Copilot; a plain `mlflow server` does not know about it.

Copilot is used only through its official CLI with the user's own GitHub login; each turn
counts as one premium request of the plan.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Literal

from mlflow.assistant.providers.base import (
    AssistantProvider,
    CLINotInstalledError,
    NotAuthenticatedError,
    load_config_or_default,
)
from mlflow.assistant.providers.claude_code import _build_system_prompt
from mlflow.assistant.types import (
    Event,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from mlflow.server.assistant.session import clear_process_pid, save_process_pid

PROVIDER_NAME = "copilot"
DEFAULT_MODEL = "gpt-5.4"
KNOWN_MODELS = ["auto", "gpt-5.4", "claude-sonnet-5"]

# Copilot CLI tool names (as the CLI reports them), grouped by what MLflow's permission
# settings mean. Reading and running commands is always allowed: the MLflow skills work by
# running `mlflow` and Python commands against the tracking server.
READ_AND_RUN_TOOLS = [
    "bash",
    "read_bash",
    "list_bash",
    "stop_bash",
    "view",
    "rg",
    "glob",
    "skill",
    "sql",
    "task",
]
EDIT_TOOLS = ["apply_patch"]
DOCS_TOOLS = ["web_fetch", "web_search", "fetch_copilot_cli_documentation"]


class CopilotProvider(AssistantProvider):
    """MLflow Assistant provider backed by the GitHub Copilot CLI."""

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return "GitHub Copilot CLI"

    @property
    def description(self) -> str:
        return (
            "Run the assistant through the GitHub Copilot CLI "
            "(your Copilot subscription; one premium request per turn)."
        )

    @property
    def client_tool_delivery(self) -> Literal["unsupported"]:
        return "unsupported"

    def is_available(self) -> bool:
        return shutil.which("copilot") is not None

    def check_connection(self, echo: Callable[[str], None] | None = None) -> None:
        copilot_path = shutil.which("copilot")
        if not copilot_path:
            if echo:
                echo("Copilot CLI not found")
            raise CLINotInstalledError(
                "GitHub Copilot CLI is not installed. Install it with: npm install -g @github/copilot"
            )
        if echo:
            echo(f"Copilot CLI found: {copilot_path}")
            echo("Checking connection... (one small request)")
        try:
            result = subprocess.run(
                [
                    copilot_path,
                    "-p",
                    "Reply with OK.",
                    "--output-format",
                    "json",
                    "--available-tools=",
                    "--no-auto-update",
                    "--no-custom-instructions",
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            raise NotAuthenticatedError("Copilot CLI did not answer within 90 seconds")
        if result.returncode == 0:
            if echo:
                echo("Authentication verified")
            return
        stderr = (result.stderr or result.stdout).strip()
        if "login" in stderr.lower() or "auth" in stderr.lower():
            raise NotAuthenticatedError(
                "Not authenticated. Run `copilot` once and use /login, or set GH_TOKEN."
            )
        raise NotAuthenticatedError(
            stderr[:500] or f"Copilot CLI exited with code {result.returncode}"
        )

    def resolve_skills_path(self, base_directory: Path) -> Path:
        # Copilot discovers project skills in .github/skills (also .claude/skills) and
        # personal skills in ~/.copilot/skills; tools/mlflow_server.py copies MLflow's there.
        return base_directory / ".github" / "skills"

    def list_models(
        self, base_url: str | None = None, api_key: str | None = None
    ) -> list[str]:
        return list(KNOWN_MODELS)

    def _build_command(
        self,
        copilot_path: str,
        config,
        session_id: str | None,
        cwd: Path | None,
        usage_file: str,
    ) -> list[str]:
        cmd = [
            copilot_path,
            "--output-format",
            "json",
            "--no-auto-update",
            "--usage-output-file",
            usage_file,
        ]
        if config.model and config.model != "default":
            cmd.extend(["--model", config.model])
        if config.permissions.full_access:
            cmd.append("--allow-all")
        else:
            allowed = list(READ_AND_RUN_TOOLS)
            if config.permissions.allow_edit_files:
                allowed.extend(EDIT_TOOLS)
            if config.permissions.allow_read_docs:
                allowed.extend(DOCS_TOOLS)
                cmd.extend(["--allow-url", "mlflow.org", "--allow-url", "github.com"])
            for tool in allowed:
                cmd.extend(["--allow-tool", tool])
        if cwd:
            cmd.extend(["--add-dir", str(cwd)])
        if session_id:
            cmd.extend(["--resume", session_id])
        return cmd

    async def astream(
        self,
        prompt: str,
        tracking_uri: str,
        session_id: str | None = None,
        mlflow_session_id: str | None = None,
        cwd: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> AsyncGenerator[Event, None]:
        copilot_path = shutil.which("copilot")
        if not copilot_path:
            yield Event.from_error(
                "Copilot CLI not found. Install it with: npm install -g @github/copilot"
            )
            return
        config = load_config_or_default(self.name)
        user_message = prompt
        if context:
            user_message = f"<context>\n{json.dumps(context)}\n</context>\n\n{prompt}"
        if not session_id:
            # The CLI has no system-prompt flag; on the first turn the MLflow system prompt
            # travels with the user message. Resumed sessions already have it.
            user_message = (
                f"<system_instructions>\n{_build_system_prompt(tracking_uri)}\n"
                f"</system_instructions>\n\n{user_message}"
            )

        fd, usage_file = tempfile.mkstemp(
            prefix="mlflow_copilot_usage_", suffix=".json"
        )
        os.close(fd)
        cmd = self._build_command(copilot_path, config, session_id, cwd, usage_file)
        cmd.extend(["-p", user_message])
        copilot_session_id = session_id
        final_text = ""
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=100 * 1024 * 1024,
                env={**os.environ.copy(), "MLFLOW_TRACKING_URI": tracking_uri},
            )
            if mlflow_session_id and process.pid:
                save_process_pid(mlflow_session_id, process.pid)
            try:
                assert process.stdout is not None
                async for raw in process.stdout:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    data = event.get("data") or {}
                    copilot_session_id = (
                        data.get("sessionId")
                        or event.get("sessionId")
                        or copilot_session_id
                    )
                    kind = event.get("type", "")
                    if kind == "assistant.message":
                        blocks: list = []
                        if data.get("content"):
                            blocks.append(TextBlock(text=data["content"]))
                            final_text = data["content"]
                        for call in data.get("toolRequests") or []:
                            blocks.append(
                                ToolUseBlock(
                                    id=call.get("toolCallId", ""),
                                    name=call.get("name", "tool"),
                                    input=call.get("arguments") or {},
                                )
                            )
                        if blocks:
                            yield Event.from_message(
                                Message(role="assistant", content=blocks)
                            )
                    elif kind == "tool.execution_complete":
                        result = data.get("result") or {}
                        content = (
                            result.get("content")
                            if isinstance(result, dict)
                            else str(result)
                        )
                        yield Event.from_message(
                            Message(
                                role="user",
                                content=[
                                    ToolResultBlock(
                                        tool_use_id=data.get("toolCallId", ""),
                                        content=content,
                                        is_error=not data.get("success", True),
                                    )
                                ],
                            )
                        )
                    elif kind == "error":
                        yield Event.from_error(
                            str(
                                data.get("message")
                                or event.get("message")
                                or "Copilot CLI error"
                            )
                        )
            finally:
                if mlflow_session_id:
                    clear_process_pid(mlflow_session_id)
            await process.wait()
            if process.returncode == -9:
                yield Event.from_interrupted()
                return
            if process.returncode != 0:
                stderr = ""
                if process.stderr:
                    stderr = (await process.stderr.read()).decode(
                        "utf-8", errors="replace"
                    )
                yield Event.from_error(
                    stderr.strip()
                    or f"Copilot CLI exited with code {process.returncode}"
                )
                return
            usage = self._read_usage(usage_file)
            if usage:
                yield Event.from_stream_event({"type": "usage", "usage": usage})
            yield Event.from_result(
                result=final_text or None, session_id=copilot_session_id or ""
            )
        except Exception as e:  # surfaced to the UI instead of crashing the stream
            yield Event.from_error(f"Copilot provider error: {e}")
        finally:
            try:
                os.unlink(usage_file)
            except OSError:
                pass

    @staticmethod
    def _read_usage(usage_file: str) -> dict[str, Any] | None:
        try:
            stats = json.loads(Path(usage_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        tokens = stats.get("tokenDetails") or {}

        def count(key):
            return int((tokens.get(key) or {}).get("tokenCount") or 0)

        prompt = count("input") + count("cache_read") + count("cache_write")
        completion = count("output")
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cache_read_tokens": count("cache_read"),
            "premium_requests": stats.get("totalPremiumRequestCost"),
        }


def install() -> None:
    """Register the provider with MLflow by wrapping its provider factories (idempotent)."""
    import mlflow.assistant.providers as providers

    if getattr(providers, "_copilot_installed", False):
        return
    original_build = providers._build_providers
    original_precedence = providers._default_provider_precedence

    def build_with_copilot():
        return [CopilotProvider(), *original_build()]

    def precedence_with_copilot(include_gateway: bool = True):
        return [
            CopilotProvider(),
            *original_precedence(include_gateway=include_gateway),
        ]

    providers._build_providers = build_with_copilot
    providers._default_provider_precedence = precedence_with_copilot
    providers._copilot_installed = True
