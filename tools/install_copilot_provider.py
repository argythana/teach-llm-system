#!/usr/bin/env python
"""Configure this machine so the MLflow Assistant defaults to the GitHub Copilot CLI.

    uv run python tools/install_copilot_provider.py                  # install (idempotent)
    uv run python tools/install_copilot_provider.py --model auto     # let Copilot pick the model
    uv run python tools/install_copilot_provider.py --remove         # undo the config entry

What it does (user-scoped, nothing inside the repository or the virtual environment):
1. Copies MLflow's assistant skills into `~/.copilot/skills/`, Copilot's personal skills
   folder, so the CLI knows how to query traces, metrics and evaluations.
2. Writes `~/.mlflow/assistant/config.json` with Copilot selected (model gpt-5.4 by
   default) and the other providers listed unselected, so they stay available in the menu.

The provider itself is registered by starting the server with
`uv run python tools/mlflow_server.py` (a plain `mlflow server` does not know it).
Requirements: the Copilot CLI on PATH (`npm install -g @github/copilot`) with a working
login.
"""

import argparse
import shutil
from pathlib import Path

COPILOT_SKILLS_DIR = Path.home() / ".copilot" / "skills"
OTHER_PROVIDERS = ["claude_code", "codex", "ollama"]


def install_skills() -> list[str]:
    import mlflow.assistant.skills as skills_pkg

    source = Path(list(skills_pkg.__path__)[0])
    COPILOT_SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    copied = []
    for skill_dir in sorted(p for p in source.iterdir() if (p / "SKILL.md").exists()):
        shutil.copytree(
            skill_dir, COPILOT_SKILLS_DIR / skill_dir.name, dirs_exist_ok=True
        )
        copied.append(skill_dir.name)
    return copied


def write_config(model: str) -> Path:
    from mlflow.assistant.config import CONFIG_PATH, AssistantConfig, ProviderConfig

    config = AssistantConfig.load()
    for name in OTHER_PROVIDERS:
        config.providers.setdefault(name, ProviderConfig(model="default"))
        config.providers[name].selected = False
    existing = config.providers.get("copilot")
    config.providers["copilot"] = ProviderConfig(
        model=model,
        selected=True,
        permissions=existing.permissions if existing else ProviderConfig().permissions,
    )
    config.save()
    return CONFIG_PATH


def remove() -> None:
    from mlflow.assistant.config import AssistantConfig

    config = AssistantConfig.load()
    if "copilot" in config.providers:
        del config.providers["copilot"]
        config.save()
        print("removed copilot from the assistant config; other providers untouched")
    if COPILOT_SKILLS_DIR.exists():
        print(f"MLflow skills left in {COPILOT_SKILLS_DIR}; delete by hand if unwanted")


def verify() -> None:
    import mlflow.assistant.providers as providers

    from llm_course.mlflow_copilot_provider import install

    install()
    names = [p.name for p in providers.list_providers()]
    default = providers.resolve_default_provider()
    print("providers seen by tools/mlflow_server.py:", names)
    print(
        "runtime default when nothing is selected:", default.name if default else None
    )
    if not shutil.which("copilot"):
        print(
            "WARNING: the `copilot` command is not on PATH; install it with: npm install -g @github/copilot"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--remove", action="store_true")
    parser.add_argument(
        "--model",
        default="gpt-5.4",
        help="Copilot model for the assistant (default gpt-5.4; 'auto' lets Copilot choose)",
    )
    args = parser.parse_args()
    if args.remove:
        remove()
        return
    skills = install_skills()
    print(f"installed {len(skills)} MLflow skills into {COPILOT_SKILLS_DIR}")
    print("wrote", write_config(args.model), f"(copilot selected, model {args.model})")
    verify()


if __name__ == "__main__":
    main()
