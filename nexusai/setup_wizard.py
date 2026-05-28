"""
First-run setup wizard for Nexus.
Runs when ~/.nexus/config.json is missing or incomplete.
"""
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich import box

from nexusai import config as _cfg

console = Console()

PRI = "#FF4500"
ACC = "#FF6B35"
YOU = "#FFB347"
ERR = "#FF2400"


def _step(n: int, total: int, title: str) -> None:
    console.print()
    console.rule(
        f"[bold {PRI}]  {title}  [/bold {PRI}][dim]  Step {n}/{total}[/dim]",
        characters="━", style=f"bold {PRI}"
    )
    console.print()


def run_wizard() -> None:
    os.system("clear")
    console.print()
    console.print(Align.center(f"[bold {PRI}] ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗[/bold {PRI}]"))
    console.print(Align.center(f"[bold {ACC}] ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝[/bold {ACC}]"))
    console.print(Align.center(f"[bold {PRI}] ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗[/bold {PRI}]"))
    console.print(Align.center(f"[bold {ACC}] ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║[/bold {ACC}]"))
    console.print(Align.center(f"[bold {PRI}] ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║[/bold {PRI}]"))
    console.print(Align.center(f"[bold {YOU}] ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝[/bold {YOU}]"))
    console.print()
    console.print(Align.center("[dim]First-run Setup  ·  Takes about 30 seconds[/dim]"))
    console.print()

    console.print(Panel(
        f"  Welcome! Before you start chatting, Nexus needs two things:\n\n"
        f"  [bold {YOU}]1.[/bold {YOU}]  Where your Ollama models live (path or default).\n"
        f"  [bold {YOU}]2.[/bold {YOU}]  Which model to use by default.\n\n"
        f"  All Nexus data will be stored in [bold white]~/.nexus/[/bold white]\n"
        f"  [dim]Sessions, knowledge base, plugins, config — all in one place.[/dim]",
        title=f"[bold {PRI}]  Welcome to Nexus  [/bold {PRI}]",
        border_style=PRI, box=box.HEAVY, padding=(1, 3),
    ))

    cfg = _cfg.get_config()

    # ── Step 1: Ollama models path ────────────────────────────────────────────
    _step(1, 3, "Ollama Models Path")

    env_path = os.environ.get("OLLAMA_MODELS", "")
    if env_path and os.path.isdir(env_path):
        console.print(
            f"  [dim]Detected[/dim] [bold {ACC}]OLLAMA_MODELS[/bold {ACC}]"
            f" [dim]env var →[/dim] [white]{env_path}[/white]"
        )
        models_path = env_path
    else:
        console.print(
            "  If your models are on an external drive or a custom directory,\n"
            "  paste the full path here. Press [bold]Enter[/bold] to use the default Ollama location.\n\n"
            f"  [dim]Example: /Volumes/MySSD/Ollama_Models[/dim]"
        )
        console.print()
        raw = console.input(
            f"[dim][[/dim] [{PRI}]Models path[/{PRI}] [{PRI}]›[/{PRI}]  "
        ).strip().strip("'\"")

        if raw and os.path.isdir(raw):
            models_path = raw
            console.print(f"\n  [bold {ACC}]✓[/bold {ACC}]  Using: [white]{models_path}[/white]\n")
        elif raw and not os.path.isdir(raw):
            console.print(f"\n  [dim]Path not found — using default Ollama location.[/dim]\n")
            models_path = ""
        else:
            models_path = ""
            console.print(f"\n  [dim]Using default Ollama models location.[/dim]\n")

    if models_path:
        cfg["ollama_models_path"] = models_path
        os.environ["OLLAMA_MODELS"] = models_path

    # ── Step 2: Discover and select default model ─────────────────────────────
    _step(2, 3, "Select Default Model")

    console.print(f"  [dim]Scanning for available models...[/dim]")
    models = _cfg.discover_models(models_path or None)

    if models:
        rows = []
        for i, (name, size) in enumerate(models, start=1):
            size_str = f"  [dim]{size}[/dim]" if size else ""
            rows.append(
                f"  [bold {YOU}]{i}[/bold {YOU}]   "
                f"[bold {PRI}]{name:<28}[/bold {PRI}]{size_str}"
            )
        rows.append(f"\n  [bold {YOU}]{len(models)+1}[/bold {YOU}]   [dim]Enter a custom model name[/dim]")

        console.print(Panel(
            "\n".join(rows),
            title=f"[bold {PRI}]  Models found on your system  [/bold {PRI}]",
            subtitle=f"[dim]  {len(models)} model(s) detected  [/dim]",
            border_style=PRI, box=box.HEAVY, padding=(1, 2),
        ))
        console.print(f"  [dim]Press Enter to use the first model in the list.[/dim]")
        console.print()

        while True:
            choice = console.input(
                f"[dim][[/dim] [{PRI}]Default model[/{PRI}] [{PRI}]›[/{PRI}]  "
            ).strip()

            if choice == "":
                default_model = models[0][0]
                break
            if choice.isdigit():
                idx = int(choice)
                if 1 <= idx <= len(models):
                    default_model = models[idx - 1][0]
                    break
                if idx == len(models) + 1:
                    default_model = console.input(
                        f"[dim][[/dim] [{PRI}]Model name[/{PRI}] [{PRI}]›[/{PRI}]  "
                    ).strip()
                    if not default_model:
                        default_model = models[0][0]
                    break
            console.print(f"  [dim]Enter a number 1–{len(models)+1}[/dim]")

    else:
        console.print(
            f"  [dim]Could not detect models automatically.[/dim]\n"
            f"  [dim]Make sure Ollama is running:[/dim] [bold {YOU}]ollama serve[/bold {YOU}]\n"
            f"  [dim]Or enter the model name manually:[/dim]"
        )
        console.print()
        default_model = console.input(
            f"[dim][[/dim] [{PRI}]Model name[/{PRI}] [{PRI}]›[/{PRI}]  "
        ).strip() or "qwen2.5:14b"

    cfg["default_model"]   = default_model
    cfg["setup_complete"]  = True
    _cfg.save_config(cfg)

    # ── Step 3: Done ──────────────────────────────────────────────────────────
    _step(3, 3, "Setup Complete")

    console.print(Panel(
        f"  [bold {ACC}]✓[/bold {ACC}]  Config saved to [white]~/.nexus/config.json[/white]\n"
        f"  [bold {ACC}]✓[/bold {ACC}]  Default model:  [bold white]{default_model}[/bold white]\n"
        f"  [bold {ACC}]✓[/bold {ACC}]  Sessions:       [white]~/.nexus/history/[/white]\n"
        f"  [bold {ACC}]✓[/bold {ACC}]  Knowledge base: [white]~/.nexus/chroma_db/[/white]\n\n"
        f"  [dim]To index your documents later, run:[/dim]\n"
        f"  [bold {YOU}]nexus-ingest /path/to/your/docs[/bold {YOU}]\n\n"
        f"  [dim]Type [/dim][bold {YOU}]/instructions[/bold {YOU}][dim] in chat for a full guide.[/dim]",
        title=f"[bold {PRI}]  You're all set!  [/bold {PRI}]",
        border_style=PRI, box=box.HEAVY, padding=(1, 3),
    ))
    console.print()
    console.input(f"  [dim]Press Enter to launch Nexus...[/dim]  ")
    os.system("clear")
