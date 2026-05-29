import os
import sys
import json
import uuid
import glob
import re
import time
import shutil
import warnings
import logging
import subprocess
import difflib
import importlib.util
from datetime import datetime

# ── Kill all log noise before any import ─────────────────────────────────────
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TOKENIZERS_PARALLELISM"]            = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]     = "1"
os.environ["TQDM_DISABLE"]                      = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HOME"] = "/Volumes/Sahith_SSD/Ollama_Models/HF_Cache"

# ── Core imports ──────────────────────────────────────────────────────────────
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.align import Align
from rich.spinner import Spinner
from rich.rule import Rule
from rich import box

console = Console()

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR         = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR        = os.path.abspath(os.path.join(APP_DIR, ".."))
HISTORY_DIR     = os.path.join(ROOT_DIR, "data", "history")
PLUGINS_DIR     = os.path.join(ROOT_DIR, "plugins")
BM25_INDEX_PATH = os.path.join(ROOT_DIR, "data", "bm25_index.pkl")
os.makedirs(HISTORY_DIR, exist_ok=True)
os.makedirs(PLUGINS_DIR, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# ── Color palette ─────────────────────────────────────────────────────────────
PRI = "#FF4500"   # OrangeRed — panels, Nexus label, borders
ACC = "#FF6B35"   # Coral — accents, success
YOU = "#FFB347"   # Amber — You label
ERR = "#FF2400"   # Scarlet — errors

# ── Constants ─────────────────────────────────────────────────────────────────
VERSION = "v3.0.0"

LOGO_LINES = [
    " ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗",
    " ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝",
    " ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗",
    " ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║",
    " ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║",
    " ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝",
]
LOGO_FINAL_COLORS = [
    f"bold {PRI}",
    f"bold {ACC}",
    f"bold {PRI}",
    f"bold {ACC}",
    f"bold {PRI}",
    f"bold {YOU}",
]

AVAILABLE_MODELS = [
    ("llama3",      "Fast   · 4.7 GB  · General purpose"),
    ("qwen2.5:14b", "Smart  · 9.0 GB  · Code & reasoning"),
    ("llama3.1",    "Latest · 4.9 GB  · Meta Llama 3.1"),
    ("gemma4:e4b",  "Google · 9.6 GB  · Gemma 4"),
]
DEFAULT_MODEL = "qwen2.5:14b"

THINK_FRAMES = [
    f"  [bold {PRI}]Nexus[/bold {PRI}]  [dim]●○○[/dim]  [dim italic]thinking...[/dim italic]",
    f"  [bold {PRI}]Nexus[/bold {PRI}]  [dim]○●○[/dim]  [dim italic]thinking...[/dim italic]",
    f"  [bold {PRI}]Nexus[/bold {PRI}]  [dim]○○●[/dim]  [dim italic]thinking...[/dim italic]",
    f"  [bold {PRI}]Nexus[/bold {PRI}]  [{ACC}]●●○[/{ACC}]  [dim italic]thinking...[/dim italic]",
    f"  [bold {PRI}]Nexus[/bold {PRI}]  [{PRI}]●●●[/{PRI}]  [dim italic]thinking...[/dim italic]",
]

# Populated at startup by load_plugins()
_loaded_plugins: dict = {}

# ── Help text (rebuilt after plugins load) ────────────────────────────────────
def build_chat_help() -> str:
    plugin_lines = ""
    if _loaded_plugins:
        plugin_lines = f"\n[bold {PRI}]PLUGINS[/bold {PRI}]\n\n"
        for name, p in _loaded_plugins.items():
            plugin_lines += f"  [bold {YOU}]/{name}[/bold {YOU}]  [dim]{p.get('description','')[:60]}[/dim]\n"

    return f"""\
[bold {PRI}]IN-CHAT COMMANDS[/bold {PRI}]

  [bold {YOU}]/help[/bold {YOU}]                    Show this command list
  [bold {YOU}]/instructions[/bold {YOU}]            Step-by-step guide for new users (8 pages)
  [bold {YOU}]/history[/bold {YOU}]                 Browse all sessions and view one
  [bold {YOU}]/rename [dim]<name>[/dim][/bold {YOU}]          Rename current session
  [bold {YOU}]/clear[/bold {YOU}]                   Clear screen and redraw header
  [bold {YOU}]/sessions[/bold {YOU}]                Return to session picker
  [bold {YOU}]/manage[/bold {YOU}]                  Rename, delete, or clone a session
  [bold {YOU}]/pr[/bold {YOU}]                      Open PR Documentation Generator
  [bold {YOU}]/model[/bold {YOU}]                   Show active model
  [bold {YOU}]/read [dim]<path>[/dim][/bold {YOU}]           View any file or directory in a panel
  [bold {YOU}]/ls [dim]<path>[/dim][/bold {YOU}]             Browse a directory tree
  [bold {YOU}]/context [dim]add|show|rm|clear[/dim][/bold {YOU}]  Pin files into every LLM call
  [bold {YOU}]/web [dim]<query>[/dim][/bold {YOU}]           Web search — inject top 3 results
  [bold {YOU}]/export[/bold {YOU}]                  Save session as Markdown to Desktop
  [bold {YOU}]/tag [dim]<label>[/dim][/bold {YOU}]           Tag this session
  [bold {YOU}]/search [dim]<keyword>[/dim][/bold {YOU}]       Search all sessions by keyword or tag
  [bold {YOU}]/diagnose [dim]<error>[/dim][/bold {YOU}]      Root-cause error analysis
  [bold {YOU}]/exit[/bold {YOU}]                    Save and exit

[bold {PRI}]FILE INJECTION[/bold {PRI}]

  [bold {YOU}]@filename[/bold {YOU}]          Workspace search — injects file into context
  [bold {YOU}]@/absolute/path[/bold {YOU}]    Injects any file anywhere on disk
  [bold {YOU}]@~/Desktop/file[/bold {YOU}]    Supports ~ home expansion
  [bold {YOU}]@/some/directory[/bold {YOU}]   Injects a directory tree listing
  [bold {YOU}]@web:query[/bold {YOU}]         Web search inline — inject results as context
  [dim]Bare paths also auto-detected: paste /path/to/file in your message[/dim]

[bold {PRI}]AGENT MODE[/bold {PRI}]

  [bold {YOU}]@agent[/bold {YOU}] [dim]<intent>[/dim]    Propose + apply file edits with approval

[bold {PRI}]SHORTCUTS[/bold {PRI}]

  [dim]After every response: [[/dim][bold {YOU}]c[/bold {YOU}][dim]] copy  [[/dim][bold {YOU}]r[/bold {YOU}][dim]] run code  Enter to continue[/dim]
  [dim]While thinking:  press [/dim][bold]ESC[/bold][dim] to interrupt[/dim]
{plugin_lines}"""


# ── Helpers ───────────────────────────────────────────────────────────────────
def relative_time(ts: float) -> str:
    if not ts: return "—"
    d = time.time() - ts
    if d < 60:     return "just now"
    if d < 3600:   return f"{int(d/60)}m ago"
    if d < 86400:  return f"{int(d/3600)}h ago"
    if d < 604800: return f"{int(d/86400)}d ago"
    return datetime.fromtimestamp(ts).strftime("%b %d")

def err(msg: str):
    console.print(f"\n  [bold {ERR}]✗  {msg}[/bold {ERR}]\n")

def ok(msg: str):
    console.print(f"\n  [bold {ACC}]✓  {msg}[/bold {ACC}]\n")

def read_key() -> str:
    """Read a single keypress without requiring Enter. Returns '' on non-TTY."""
    import tty, termios, select
    if not sys.stdin.isatty():
        return ""
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ── Session data ──────────────────────────────────────────────────────────────
class ChatSession:
    def __init__(self, session_id=None, model=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.model      = model or DEFAULT_MODEL
        self.created_at = time.time()
        self.updated_at = time.time()
        self.filepath   = os.path.join(HISTORY_DIR, f"session_{self.session_id}.json")
        self.history    = []
        self.pinned     = []   # list of absolute paths pinned for every call
        self.tags       = []   # list of string labels

        self.expert_system = SystemMessage(content=(
            "You are an IDE Product Expert Agent — a fully local, privacy-first AI that answers "
            "questions based ONLY on the provided Background Context from the user's knowledge base. "
            "If the answer is not in the context, say so clearly. Do NOT hallucinate."
        ))
        self.generic_system = SystemMessage(content=(
            "You are a brilliant world-class Senior Software Engineer and AI Assistant. "
            "Answer with stunning accuracy, write optimized code, and brainstorm freely "
            "using your full pre-trained knowledge."
        ))

    def load(self):
        if not os.path.exists(self.filepath): return
        try:
            with open(self.filepath) as f: data = json.load(f)
            msgs = data if isinstance(data, list) else data.get("messages", [])
            if not isinstance(data, list):
                self.model      = data.get("model", self.model)
                self.created_at = data.get("created_at", self.created_at)
                self.updated_at = data.get("updated_at", self.updated_at)
                self.pinned     = data.get("pinned", [])
                self.tags       = data.get("tags", [])
            for m in msgs:
                if m["type"] == "human": self.history.append(HumanMessage(content=m["content"]))
                elif m["type"] == "ai":  self.history.append(AIMessage(content=m["content"]))
        except Exception:
            err("Session file corrupted — starting fresh.")
            self.history = []

    def save(self):
        self.updated_at = time.time()
        msgs = []
        for m in self.history:
            if isinstance(m, HumanMessage): msgs.append({"type": "human", "content": m.content})
            elif isinstance(m, AIMessage):  msgs.append({"type": "ai",    "content": m.content})
        with open(self.filepath, "w") as f:
            json.dump({
                "model": self.model, "created_at": self.created_at,
                "updated_at": self.updated_at, "messages": msgs,
                "pinned": self.pinned, "tags": self.tags,
            }, f, indent=2)

    def rename(self, new_id: str):
        old = self.filepath
        self.session_id = new_id
        self.filepath   = os.path.join(HISTORY_DIR, f"session_{new_id}.json")
        if os.path.exists(old): os.rename(old, self.filepath)

    def print_conversation(self):
        if not self.history:
            console.print("  [dim]No messages in this session.[/dim]"); return
        console.print()
        console.rule(f"[{PRI}]  {self.session_id.replace('_',' ')}  [/{PRI}]", characters="━")
        for m in self.history:
            if isinstance(m, HumanMessage):
                console.print()
                console.print(f"[dim][[/dim] [bold {YOU}]You[/bold {YOU}] [{PRI}]›[/{PRI}]  {m.content}")
            else:
                console.print()
                console.print(f"  [bold {PRI}]Nexus ›[/bold {PRI}]")
                console.print()
                console.print(Markdown(m.content, code_theme="monokai"))
        console.print()
        console.rule(characters="━", style="dim")
        console.print()


def list_sessions():
    out = []
    for fp in glob.glob(os.path.join(HISTORY_DIR, "session_*.json")):
        sid = os.path.basename(fp).replace("session_", "").replace(".json", "")
        try:
            with open(fp) as f: data = json.load(f)
            if isinstance(data, list):
                count, updated, model, tags = len(data)//2, os.path.getmtime(fp), "llama3", []
            else:
                count   = len(data.get("messages", []))//2
                updated = data.get("updated_at", os.path.getmtime(fp))
                model   = data.get("model", "llama3")
                tags    = data.get("tags", [])
            out.append((sid, count, updated, model, tags))
        except Exception: pass
    out.sort(key=lambda x: x[2], reverse=True)
    return out


# ── Splash with pulse animation ───────────────────────────────────────────────
def render_splash(animated=True):
    os.system("clear")
    console.print()

    if animated:
        for line in LOGO_LINES:
            console.print(f"[bold bright_white]{line}[/bold bright_white]", justify="center")
        time.sleep(0.18)
        os.system("clear")
        console.print()
        for line, color in zip(LOGO_LINES, LOGO_FINAL_COLORS):
            console.print(f"[{color}]{line}[/{color}]", justify="center")
            time.sleep(0.07)
    else:
        for line, color in zip(LOGO_LINES, LOGO_FINAL_COLORS):
            console.print(f"[{color}]{line}[/{color}]", justify="center")

    console.print()
    console.print(Align.center(
        "[dim]Local Offline IDE Expert Agent  ·  Privacy-First  ·  Zero Telemetry[/dim]"
    ))
    console.print(Align.center(
        f"[bold dim]{VERSION}[/bold dim]  [dim]·  Offline Mode[/dim]"
    ))
    console.print()
    console.rule(characters="━", style=f"bold {PRI}")
    console.print()
    console.print(Align.center(
        "[dim]In chat →[/dim]  "
        f"[{YOU}]/help[/{YOU}]  [dim]·[/dim]  "
        f"[bold {YOU}]/instructions[/bold {YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/history[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/context[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/web[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/export[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/tag[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/search[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/diagnose[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/pr[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/exit[/{YOU}]"
    ))
    console.print(Align.center(
        "[dim]Files →[/dim]  "
        f"[{YOU}]@/abs/path[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]@~/path[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]@/directory[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]@web:query[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/read <path>[/{YOU}]  [dim]·[/dim]  "
        f"[{YOU}]/ls <path>[/{YOU}]  [dim]·  paste any /path to auto-inject[/dim]"
    ))
    console.print(Align.center(
        f"[dim]Agent mode → type [/dim][bold {YOU}]@agent[/bold {YOU}] [dim]<intent>  ·  "
        f"Press [bold]ESC[/bold] while thinking to interrupt[/dim]"
    ))
    console.print()


# ── Session picker ─────────────────────────────────────────────────────────────
def _session_panel_body(sessions):
    col  = 36
    rows = []
    for i, (sid, count, updated, _, tags) in enumerate(sessions, start=1):
        display = sid.replace("_", " ")
        if len(display) > col: display = display[:col-1] + "…"
        tag_str = "  " + "  ".join(f"[{ACC}]#{t}[/{ACC}]" for t in tags[:3]) if tags else ""
        rows.append(
            f"  [bold {YOU}]{i}[/bold {YOU}]   "
            f"[white]{display:<{col}}[/white]  "
            f"[dim]{relative_time(updated)}[/dim]"
            f"  [dim]{count} msg{'s' if count!=1 else ''}[/dim]"
            f"{tag_str}"
        )
    if sessions: rows.append("")
    rows.append(f"  [dim]0   Start a new conversation[/dim]")
    return "\n".join(rows)

def render_session_menu():
    sessions = list_sessions()
    n        = len(sessions)

    console.print(Panel(
        _session_panel_body(sessions),
        title=f"[bold {PRI}]  Continue a conversation  [/bold {PRI}]",
        border_style=PRI, box=box.HEAVY, padding=(1, 2),
    ))
    prompt_hint = (
        f"  [dim]Enter 1–{n} to load · 0 for new"
        + (" · /manage to organize" if n > 0 else "")
        + "[/dim]"
    ) if n else "  [dim]No sessions yet — press Enter or 0 to start.[/dim]"
    console.print(prompt_hint)
    console.print()

    while True:
        choice = console.input(f"[dim][[/dim] [{PRI}]Session[/{PRI}] [{PRI}]›[/{PRI}]  ").strip()

        if choice in ("0", "", "N", "n"):
            return None
        if choice.lower() == "p":
            return "__pr__"
        if choice.lower() in ("/manage", "manage"):
            result = _manage_sessions_inline(sessions)
            if result == "__refresh__":
                sessions = list_sessions(); n = len(sessions)
                console.print()
                console.print(Panel(_session_panel_body(sessions),
                    title=f"[bold {PRI}]  Continue a conversation  [/bold {PRI}]",
                    border_style=PRI, box=box.HEAVY, padding=(1, 2)))
                console.print(prompt_hint); console.print()
            elif result is not None:
                return result
            continue

        parts = choice.split()
        if len(parts) == 1 and parts[0].isdigit():
            idx = int(parts[0]) - 1
            if 0 <= idx < n:
                return sessions[idx][0]
            err(f"Enter a number between 1 and {n}, or 0 for new.")
            continue

        err("Not recognised — type a number, 0 for new, or /manage.")

def _manage_sessions_inline(sessions):
    n = len(sessions)
    if n == 0:
        err("No sessions to manage yet."); return None

    rows = "\n".join(
        f"  [{YOU}]{i+1}[/{YOU}]  [white]{s[0].replace('_',' ')}[/white]"
        for i, s in enumerate(sessions)
    )
    body = (
        f"[dim]Sessions:[/dim]\n\n{rows}\n\n"
        f"[bold {YOU}]d N[/bold {YOU}]          [dim]Delete session #N[/dim]\n"
        f"[bold {YOU}]r N newname[/bold {YOU}]  [dim]Rename session #N[/dim]\n"
        f"[bold {YOU}]b N newname[/bold {YOU}]  [dim]Clone session #N as 'newname'[/dim]\n\n"
        "[dim]Press Enter to cancel[/dim]"
    )
    console.print()
    console.print(Panel(body, title=f"[bold {PRI}]  Manage Sessions  [/bold {PRI}]",
                        border_style=PRI, box=box.HEAVY, padding=(1, 2)))
    console.print()
    action = console.input(f"[dim][[/dim] [{PRI}]Action[/{PRI}] [{PRI}]›[/{PRI}]  ").strip()

    if not action: console.print("  [dim]Cancelled.[/dim]"); return None

    parts = action.split()

    if len(parts) == 2 and parts[0].lower() == "d" and parts[1].isdigit():
        idx = int(parts[1]) - 1
        if 0 <= idx < n:
            sid = sessions[idx][0]
            os.remove(os.path.join(HISTORY_DIR, f"session_{sid}.json"))
            ok(f"Deleted: {sid.replace('_',' ')}"); return "__refresh__"
        err(f"No session #{parts[1]}."); return None

    if len(parts) >= 3 and parts[0].lower() == "r" and parts[1].isdigit():
        idx = int(parts[1]) - 1; new_id = "_".join(parts[2:]).lower()
        if 0 <= idx < n:
            old = sessions[idx][0]
            os.rename(os.path.join(HISTORY_DIR, f"session_{old}.json"),
                      os.path.join(HISTORY_DIR, f"session_{new_id}.json"))
            ok(f"Renamed: {old.replace('_',' ')}  →  {new_id.replace('_',' ')}"); return "__refresh__"
        err(f"No session #{parts[1]}."); return None

    if len(parts) >= 3 and parts[0].lower() == "b" and parts[1].isdigit():
        idx = int(parts[1]) - 1; new_id = "_".join(parts[2:]).lower()
        if 0 <= idx < n:
            old = sessions[idx][0]
            shutil.copy2(os.path.join(HISTORY_DIR, f"session_{old}.json"),
                         os.path.join(HISTORY_DIR, f"session_{new_id}.json"))
            ok(f"Cloned: {old.replace('_',' ')}  →  {new_id.replace('_',' ')}"); return new_id
        err(f"No session #{parts[1]}."); return None

    err("Unknown action — use  d N · r N name · b N name  or Enter."); return None


# ── Model selector ─────────────────────────────────────────────────────────────
def render_model_selector():
    rows = []
    for i, (model, desc) in enumerate(AVAILABLE_MODELS, start=1):
        marker = f"  [bold {ACC}]← default[/bold {ACC}]" if model == DEFAULT_MODEL else ""
        rows.append(
            f"  [bold {YOU}]{i}[/bold {YOU}]   "
            f"[bold {PRI}]{model:<16}[/bold {PRI}]  "
            f"[dim]{desc}[/dim]{marker}"
        )
    console.print()
    console.print(Panel("\n".join(rows),
        title=f"[bold {PRI}]  Select a model  [/bold {PRI}]",
        border_style=PRI, box=box.HEAVY, padding=(1, 2)))
    console.print(f"  [dim]Enter 1–{len(AVAILABLE_MODELS)}, or press Enter for {DEFAULT_MODEL}[/dim]")
    console.print()

    while True:
        choice = console.input(f"[dim][[/dim] [{PRI}]Model[/{PRI}] [{PRI}]›[/{PRI}]  ").strip()
        if choice == "": return DEFAULT_MODEL
        if choice.isdigit() and 1 <= int(choice) <= len(AVAILABLE_MODELS):
            return AVAILABLE_MODELS[int(choice)-1][0]
        err(f"Enter a number 1–{len(AVAILABLE_MODELS)} or press Enter.")


# ── Status bar ─────────────────────────────────────────────────────────────────
def render_status_bar(session, git_branch: str = ""):
    exchanges = len([m for m in session.history if isinstance(m, HumanMessage)])
    name      = session.session_id.replace("_", " ")
    pinned_str = (
        f"  [dim]·  Pinned[/dim] [bold {ACC}]{len(session.pinned)}[/bold {ACC}]"
        if session.pinned else ""
    )
    tag_str = (
        "  [dim]·[/dim]  " + "  ".join(f"[{ACC}]#{t}[/{ACC}]" for t in session.tags[:3])
        if session.tags else ""
    )
    git_str = (
        f"  [dim]·  git:[/dim] [bold white]{git_branch}[/bold white]"
        if git_branch else ""
    )
    console.print()
    console.rule(characters="━", style=f"bold {PRI}")
    console.print(
        f"  [dim]Session[/dim] [bold white]{name}[/bold white]  "
        f"[dim]·  Model[/dim] [bold {ACC}]{session.model}[/bold {ACC}]  "
        f"[dim]·  Exchanges[/dim] [bold white]{exchanges}[/bold white]"
        f"{git_str}{pinned_str}{tag_str}  "
        f"[dim]·  ●[/dim] [dim {PRI}]Offline[/dim {PRI}]"
    )
    console.rule(characters="━", style=f"bold {PRI}")
    console.print()


# ── /history command ──────────────────────────────────────────────────────────
def cmd_history():
    sessions = list_sessions()
    if not sessions:
        console.print("  [dim]No saved sessions.[/dim]"); return

    rows = "\n".join(
        f"  [bold {YOU}]{i+1}[/bold {YOU}]   "
        f"[white]{s[0].replace('_',' '):<40}[/white]  "
        f"[dim]{relative_time(s[2])}[/dim]  "
        f"[dim]{s[1]} msg{'s' if s[1]!=1 else ''}[/dim]"
        + ("  " + "  ".join(f"[{ACC}]#{t}[/{ACC}]" for t in s[4][:2]) if s[4] else "")
        for i, s in enumerate(sessions)
    )
    console.print()
    console.print(Panel(rows,
        title=f"[bold {PRI}]  Session History  [/bold {PRI}]",
        subtitle="[dim]  type a number to view · Enter to cancel  [/dim]",
        border_style=PRI, box=box.HEAVY, padding=(1, 2)))
    console.print()
    choice = console.input(f"[dim][[/dim] [{PRI}]View #[/{PRI}] [{PRI}]›[/{PRI}]  ").strip()

    if not choice: console.print("  [dim]Cancelled.[/dim]\n"); return
    if choice.isdigit() and 1 <= int(choice) <= len(sessions):
        idx = int(choice) - 1
        sid, _, _, model, _ = sessions[idx]
        tmp = ChatSession(session_id=sid, model=model)
        tmp.load(); tmp.print_conversation()
    else:
        err(f"No session #{choice}.")


# ── PR generator ──────────────────────────────────────────────────────────────
def handle_pr_generator():
    if ROOT_DIR not in sys.path: sys.path.insert(0, ROOT_DIR)
    try:
        from pr_generator.generator import run
        run(os.path.join(ROOT_DIR, "data"))
    except ImportError as e:
        err(f"PR generator unavailable: {e}")
    console.print()
    console.rule(f"[{PRI}]  back in chat  [/{PRI}]", characters="━")
    console.print()


# ── Workspace helpers ─────────────────────────────────────────────────────────
def generate_tree_map(startpath):
    _skip = {".git", "node_modules", "venv", "__pycache__", "mac", "data"}
    tree  = []
    for root, dirs, files in os.walk(startpath):
        level  = root.replace(startpath, "").count(os.sep)
        indent = "    " * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            if not f.startswith("."): tree.append(f"{'    '*(level+1)}{f}")
        dirs[:] = [d for d in dirs if d not in _skip]
    return "\n".join(tree[:200])

def find_custom_file(filename, startpath):
    _skip = {".git", "node_modules", "venv", "__pycache__", "mac", "data"}
    for root, dirs, files in os.walk(startpath):
        if filename in files: return os.path.join(root, filename)
        dirs[:] = [d for d in dirs if d not in _skip]
    return None

def read_any_path(path: str) -> str:
    """Read any file to a string. Handles PDFs, text, and binary gracefully."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        try:
            import fitz
            doc   = fitz.open(path)
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    pages.append(f"[Page {i+1}]\n{text}")
            if pages:
                return "\n\n".join(pages[:15])   # cap at 15 pages
            return "[PDF has no extractable text — may be a scanned image PDF]"
        except Exception as e:
            return f"[PDF: {os.path.basename(path)}, extraction failed: {e}]"

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        size = os.path.getsize(path)
        return f"[Binary file: {ext or 'unknown'}, {size:,} bytes — cannot read as text]"


# ── F7: Git-aware context ─────────────────────────────────────────────────────
def get_git_context(cwd: str) -> tuple:
    """Returns (branch, context_string). Empty strings if not a git repo."""
    try:
        r = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                           capture_output=True, text=True, cwd=cwd, timeout=5)
        if r.returncode != 0:
            return "", ""

        branch = subprocess.run(["git", "branch", "--show-current"],
                                 capture_output=True, text=True, cwd=cwd, timeout=5
                                 ).stdout.strip()

        log = subprocess.run(["git", "log", "--oneline", "-8"],
                              capture_output=True, text=True, cwd=cwd, timeout=5
                              ).stdout.strip()

        stat = subprocess.run(["git", "diff", "--stat", "HEAD"],
                               capture_output=True, text=True, cwd=cwd, timeout=5
                               ).stdout.strip()

        patch = subprocess.run(["git", "diff", "--unified=2"],
                                capture_output=True, text=True, cwd=cwd, timeout=5
                                ).stdout.strip()
        if len(patch) > 3000:
            patch = patch[:3000] + "\n...(diff truncated)"

        parts = [f"Branch: {branch}"]
        if log:   parts.append(f"Recent commits:\n{log}")
        if stat:  parts.append(f"Uncommitted changes:\n{stat}")
        if patch: parts.append(f"Diff:\n{patch}")

        return branch, "\n\n".join(parts)
    except Exception:
        return "", ""


# ── F2: Web search ────────────────────────────────────────────────────────────
def web_search(query: str) -> str:
    """Fetch top 3 DuckDuckGo results and return formatted string for LLM injection."""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "Web search unavailable — run: pip install ddgs"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"Web search for '{query}': no results found."
        lines = [f"Web search results for: {query}\n"]
        for i, r in enumerate(results, 1):
            lines.append(
                f"[{i}] {r.get('title','')}\n"
                f"URL: {r.get('href','')}\n"
                f"{r.get('body','')}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return (
            f"Web search failed for '{query}'.\n"
            f"Reason: {e}\n"
            f"Tip: check your internet connection, or try again in a moment."
        )


# ── F3+F4+F6: Post-response helpers ──────────────────────────────────────────
def extract_code_blocks(text: str) -> list:
    """Returns list of (lang, code) tuples from fenced code blocks."""
    return [
        (lang.strip(), code.strip())
        for lang, code in re.findall(r'```(\w*)\n(.*?)```', text, re.DOTALL)
        if code.strip()
    ]

def _execute_block(lang: str, code: str) -> None:
    cmd_map = {
        "python": [sys.executable, "-c", code],
        "py":     [sys.executable, "-c", code],
        "bash":   ["bash", "-c", code],
        "sh":     ["bash", "-c", code],
        "js":     ["node", "-e", code],
        "javascript": ["node", "-e", code],
        "":       ["bash", "-c", code],
    }
    cmd = cmd_map.get(lang.lower(), ["bash", "-c", code])

    console.print(Panel(
        Markdown(f"```{lang}\n{code}\n```", code_theme="monokai"),
        title=f"[bold {PRI}]  Run {lang or 'code'}?  [/bold {PRI}]",
        border_style=f"dim {PRI}", box=box.HEAVY, padding=(1, 2)
    ))
    confirm = console.input(
        f"  [dim][[/dim][{YOU}]y[/{YOU}][dim]/N] ›  [/dim]"
    ).strip().lower()
    if confirm != "y":
        console.print("  [dim]Cancelled.[/dim]\n"); return

    with Live(Spinner("dots", style=PRI, text=f"  [{PRI}]Running...[/{PRI}]"),
              refresh_per_second=12, transient=True):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=30, cwd=os.getcwd())
        except subprocess.TimeoutExpired:
            err("Timed out after 30 seconds."); return
        except FileNotFoundError as e:
            err(f"Interpreter not found: {e}"); return

    output  = result.stdout
    if result.stderr: output += f"\n[stderr]\n{result.stderr}"
    status  = f"[{ACC}]✓ exit 0[/{ACC}]" if result.returncode == 0 else f"[{ERR}]✗ exit {result.returncode}[/{ERR}]"
    console.print(Panel(
        output.strip() or "[dim](no output)[/dim]",
        title=f"[bold {PRI}]  Output  [/bold {PRI}]",
        subtitle=f"  {status}  ",
        border_style=PRI, box=box.HEAVY, padding=(1, 2)
    ))

def post_response_prompt(ai_response: str) -> None:
    """After a response: offer copy / run / continue."""
    blocks   = extract_code_blocks(ai_response)
    runnable = [(l, c) for l, c in blocks
                if l.lower() in ("python","py","bash","sh","js","javascript","")]
    has_run  = bool(runnable)

    hint = (
        f"  [dim][[/dim] [{YOU}]c[/{YOU}] [dim]copy[/dim]"
        + (f"  [{YOU}]r[/{YOU}] [dim]run[/dim]" if has_run else "")
        + f"  [dim]Enter to continue ][/dim]"
    )
    console.print(hint, end="")
    sys.stdout.flush()
    ch = read_key()
    console.print()

    if ch.lower() == "c":
        try:
            import pyperclip
            pyperclip.copy(ai_response)
            ok("Copied to clipboard.")
        except Exception as e:
            err(f"Copy failed: {e}")

    elif ch.lower() == "r" and has_run:
        if len(runnable) == 1:
            _execute_block(*runnable[0])
        else:
            rows = "\n".join(
                f"  [{YOU}]{i+1}[/{YOU}]  [dim]{lang or 'code'}[/dim]  "
                f"[white]{code[:70].replace(chr(10),' ')}…[/white]"
                for i, (lang, code) in enumerate(runnable)
            )
            console.print(Panel(rows, title=f"[bold {PRI}]  Select block to run  [/bold {PRI}]",
                                border_style=PRI, box=box.HEAVY, padding=(1, 2)))
            pick = console.input(f"[dim][[/dim] [{PRI}]#[/{PRI}] [{PRI}]›[/{PRI}]  ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(runnable):
                _execute_block(*runnable[int(pick)-1])


# ── F5: /export ───────────────────────────────────────────────────────────────
def cmd_export(session) -> None:
    if not session.history:
        err("Nothing to export — session is empty."); return

    lines = [
        f"# Nexus Session: {session.session_id.replace('_', ' ')}\n",
        f"**Model:** {session.model}  |  **Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
    ]
    if session.tags:
        lines.append("**Tags:** " + "  ".join(f"#{t}" for t in session.tags) + "\n")
    lines.append("---\n")

    for m in session.history:
        if isinstance(m, HumanMessage):
            lines.append(f"\n## You\n\n{m.content}\n")
        elif isinstance(m, AIMessage):
            lines.append(f"\n## Nexus\n\n{m.content}\n")

    filename = f"nexus_{session.session_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    desktop  = os.path.join(os.path.expanduser("~"), "Desktop", filename)
    try:
        with open(desktop, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        ok(f"Exported: {desktop}")
    except Exception as e:
        err(f"Export failed: {e}")


# ── F9: Session tags + search ─────────────────────────────────────────────────
def cmd_tag(session, label: str) -> None:
    if not label:
        if session.tags:
            console.print(
                "\n  [dim]Tags:[/dim]  "
                + "  ".join(f"[{ACC}]#{t}[/{ACC}]" for t in session.tags)
                + "\n"
            )
        else:
            console.print("  [dim]No tags. Usage: /tag <label>[/dim]\n")
        return
    tag = label.strip().lower().replace(" ", "_")
    if tag not in session.tags:
        session.tags.append(tag)
        session.save()
        ok(f"Tagged: #{tag}")
    else:
        console.print(f"  [dim]Already tagged #{tag}[/dim]\n")

def cmd_search(keyword: str) -> None:
    if not keyword:
        err("Usage: /search <keyword>"); return
    kw      = keyword.lower()
    matches = []

    for fp in glob.glob(os.path.join(HISTORY_DIR, "session_*.json")):
        try:
            with open(fp) as f: data = json.load(f)
            msgs = data if isinstance(data, list) else data.get("messages", [])
            sid  = os.path.basename(fp).replace("session_","").replace(".json","")
            tags = data.get("tags", []) if not isinstance(data, list) else []

            snippets = []
            for m in msgs:
                if kw in m.get("content","").lower():
                    snip = m["content"][:100].replace("\n"," ")
                    snippets.append(f"  [dim]{m['type']}:[/dim] {snip}…")

            if snippets or any(kw in t for t in tags):
                matches.append((sid, tags, snippets[:2]))
        except Exception:
            pass

    if not matches:
        console.print(f"\n  [dim]No results for «{keyword}»[/dim]\n"); return

    rows = []
    for sid, tags, snippets in matches:
        tag_str = "  " + "  ".join(f"[{ACC}]#{t}[/{ACC}]" for t in tags) if tags else ""
        rows.append(f"\n  [bold {YOU}]{sid.replace('_',' ')}[/bold {YOU}]{tag_str}")
        rows.extend(snippets)

    console.print(Panel(
        "\n".join(rows),
        title=f"[bold {PRI}]  Search: {keyword}  ·  {len(matches)} session(s)  [/bold {PRI}]",
        border_style=PRI, box=box.HEAVY, padding=(1, 2)
    ))


# ── F1: Pinned context ────────────────────────────────────────────────────────
def cmd_context(session, arg: str) -> None:
    parts = arg.strip().split(maxsplit=1) if arg else []
    sub   = parts[0].lower() if parts else "show"
    val   = parts[1].strip() if len(parts) > 1 else ""

    if sub in ("show", "") or not parts:
        if not session.pinned:
            console.print("  [dim]No pinned files. Use /context add <path>[/dim]\n"); return
        rows = "\n".join(
            f"  [{YOU}]{i+1}[/{YOU}]  [white]{p}[/white]"
            for i, p in enumerate(session.pinned)
        )
        console.print(Panel(rows,
            title=f"[bold {PRI}]  Pinned Context ({len(session.pinned)} file(s))  [/bold {PRI}]",
            border_style=PRI, box=box.HEAVY, padding=(1, 2)))

    elif sub == "add":
        if not val: err("Usage: /context add <path>"); return
        rp = os.path.realpath(os.path.expanduser(val))
        if not os.path.exists(rp): err(f"Path not found: {rp}"); return
        if rp not in session.pinned:
            session.pinned.append(rp)
            session.save()
            ok(f"Pinned: {os.path.basename(rp)}")
        else:
            console.print("  [dim]Already pinned.[/dim]\n")

    elif sub in ("rm", "remove"):
        if not val or not val.isdigit(): err("Usage: /context rm <number>"); return
        idx = int(val) - 1
        if 0 <= idx < len(session.pinned):
            removed = session.pinned.pop(idx)
            session.save()
            ok(f"Unpinned: {os.path.basename(removed)}")
        else:
            err(f"No item #{val}")

    elif sub == "clear":
        session.pinned.clear()
        session.save()
        ok("Pinned context cleared.")

    else:
        err("Usage: /context [add <path> | show | rm <n> | clear]")

def build_pinned_context(pinned: list) -> str:
    result = ""
    for path in pinned:
        if not os.path.exists(path): continue
        try:
            if os.path.isdir(path):
                tree = generate_tree_map(path)
                result += f"\n--- Pinned Directory: {path} ---\n{tree}\n"
            else:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                result += f"\n--- Pinned File: {path} ---\n{content}\n"
        except Exception:
            pass
    return result


# ── F8: /diagnose ─────────────────────────────────────────────────────────────
def cmd_diagnose(error_msg: str, session, llm, root_dir: str) -> None:
    if not error_msg:
        err("Usage: /diagnose <error message or stack trace>"); return

    console.print(f"\n  [dim {ACC}]Scanning workspace for relevant files...[/dim {ACC}]")

    file_refs = re.findall(r'"([^"]+\.py)"', error_msg) + \
                re.findall(r'(?:File\s+")([^"]+)"', error_msg) + \
                re.findall(r'([\w/.\-]+\.py)', error_msg)
    symbols   = re.findall(r'\b([A-Z][A-Za-z]+(?:Error|Exception|Warning))\b', error_msg) + \
                re.findall(r'in ([a-z_]\w+)\b', error_msg)

    injected = ""
    seen     = set()

    for ref in dict.fromkeys(file_refs[:5]):
        fp = find_custom_file(os.path.basename(ref), root_dir) or \
             (ref if os.path.isabs(ref) and os.path.exists(ref) else None)
        if fp and fp not in seen:
            seen.add(fp)
            try:
                with open(fp, "r", encoding="utf-8") as f: code = f.read()
                injected += f"\n--- File: {fp} ---\n{code[:3000]}\n"
            except Exception: pass

    if len(injected) < 6000:
        for sym in dict.fromkeys(symbols[:6]):
            if len(sym) < 3: continue
            try:
                r = subprocess.run(["grep", "-r", "--include=*.py", "-l", sym, root_dir],
                                   capture_output=True, text=True, timeout=5)
                for fp in r.stdout.strip().split("\n")[:3]:
                    if fp and fp not in seen:
                        seen.add(fp)
                        try:
                            with open(fp, "r", encoding="utf-8") as f: code = f.read()
                            injected += f"\n--- File: {fp} ---\n{code[:2000]}\n"
                        except Exception: pass
            except Exception: pass

    if seen:
        console.print(f"  [dim]Found {len(seen)} relevant file(s). Diagnosing...[/dim]\n")
    else:
        console.print(f"  [dim]No specific files found — running general diagnosis.[/dim]\n")

    diag_prompt = (
        "You are an expert debugger. Diagnose the following error with precision.\n"
        "Structure your response:\n"
        "1. **Root Cause** — the real reason this error occurs\n"
        "2. **Location** — exact file and line if determinable\n"
        "3. **Fix** — concrete code that resolves it\n"
        "4. **Why it happened** — brief explanation\n\n"
        f"=== ERROR ===\n{error_msg}\n=============\n"
    )
    if injected:
        diag_prompt += f"\n=== RELEVANT FILES ===\n{injected}\n=====================\n"
    diag_prompt += "\nDiagnosis:"

    ai_response = stream_response(llm, diag_prompt)
    session.history.append(HumanMessage(content=f"/diagnose {error_msg}"))
    session.history.append(AIMessage(content=ai_response))
    session.save()
    post_response_prompt(ai_response)


# ── F11: Plugin system ────────────────────────────────────────────────────────
def load_plugins() -> None:
    global _loaded_plugins
    if not os.path.isdir(PLUGINS_DIR):
        return
    for fp in sorted(glob.glob(os.path.join(PLUGINS_DIR, "*.py"))):
        try:
            spec = importlib.util.spec_from_file_location("nexus_plugin", fp)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            plugin = getattr(mod, "PLUGIN", None)
            if plugin and "name" in plugin and "run" in plugin:
                name = plugin["name"].lower().strip()
                _loaded_plugins[name] = plugin
        except Exception as e:
            console.print(
                f"  [dim red]Plugin load failed ({os.path.basename(fp)}): {e}[/dim red]"
            )


# ── Prompt builder ────────────────────────────────────────────────────────────
def extract_prompt(messages, context_text="", injected_code="",
                   tree_map="", git_context="", pinned_context=""):
    prompt = messages[0].content + "\n\n"
    if tree_map:
        prompt += f"=== WORKSPACE DIRECTORY MAP ===\n{tree_map}\n================================\n\n"
    if git_context:
        prompt += f"=== GIT REPOSITORY STATE ===\n{git_context}\n============================\n\n"
    if pinned_context:
        prompt += f"=== PINNED FILES (always in context) ===\n{pinned_context}\n========================================\n\n"
    for m in messages[1:-1]:
        if isinstance(m, HumanMessage): prompt += f"Human: {m.content}\n"
        elif isinstance(m, AIMessage):  prompt += f"Agent: {m.content}\n"
    if context_text:
        prompt += f"=== RELEVANT BACKGROUND CONTEXT ===\n{context_text}\n===================================\n\n"
    if injected_code:
        prompt += f"=== INJECTED SOURCE CODE FILES ===\n{injected_code}\n==================================\n\n"
    prompt += f"Human: {messages[-1].content}\nAgent: "
    return prompt


# ── LangGraph ─────────────────────────────────────────────────────────────────
def build_langgraph(root_dir):
    from semantic_router import Route, SemanticRouter
    from semantic_router.encoders import HuggingFaceEncoder
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings
    import pickle
    from langgraph.graph import StateGraph, END
    from typing import TypedDict

    class G(TypedDict):
        question: str; route_name: str; context_text: str

    encoder = HuggingFaceEncoder(name="mixedbread-ai/mxbai-embed-large-v1")
    routes  = [
        Route(name="generic",          utterances=["hello","hi","how are you","what is python","write a script"]),
        Route(name="code",             utterances=["find the bug","explain this class","where is the login function"]),
        Route(name="architecture",     utterances=["system design","how does it scale","architecture overview"]),
        Route(name="business_logic",   utterances=["business rules","how does billing work","compliance"]),
        Route(name="ops",              utterances=["configure the server","deployment pipeline","kubernetes"]),
        Route(name="tribal_knowledge", utterances=["why did we choose this","who owns the repo","read the docs"]),
    ]
    router = SemanticRouter(encoder=encoder, routes=routes, auto_sync="local")

    chroma_d = os.path.join(root_dir, "data", "chroma_db")
    vdb      = Chroma(persist_directory=chroma_d,
                      embedding_function=OllamaEmbeddings(model="nomic-embed-text")) \
               if os.path.exists(chroma_d) else None

    bm25 = None
    if os.path.exists(BM25_INDEX_PATH):
        try:
            with open(BM25_INDEX_PATH, "rb") as f: bm25 = pickle.load(f)
            bm25.k = 2
        except Exception: pass

    def classify(state: G):
        r = router(state["question"])
        return {"route_name": getattr(r, "name", "generic")}

    def retrieve(state: G):
        c = ""
        try:
            cr = vdb.max_marginal_relevance_search(state["question"], k=4, fetch_k=20) if vdb else []
            br = bm25.invoke(state["question"]) if bm25 else []
            seen, unique = set(), []
            for d in cr + br:
                k = d.page_content + d.metadata.get("source","")
                if k not in seen: seen.add(k); unique.append(d)
            snips = []
            for d in unique[:6]:
                src = os.path.basename(d.metadata.get("source","Unknown"))
                cit = ", ".join(filter(None,[
                    f"Page {d.metadata['page']}" if "page" in d.metadata else "",
                    f"Slide {d.metadata['slide']}" if "slide" in d.metadata else "",
                    f"Sheet {d.metadata['sheet']}" if "sheet" in d.metadata else ""]))
                snips.append(f"--- {src}{' ('+cit+')' if cit else ''} ---\n{d.page_content}")
            c = "\n\n".join(snips)
        except Exception: pass
        return {"context_text": c}

    def bypass(state: G): return {"context_text": ""}

    wf = StateGraph(G)
    wf.add_node("classify", classify)
    wf.add_node("retrieve", retrieve)
    wf.add_node("bypass",   bypass)
    wf.set_entry_point("classify")
    wf.add_conditional_edges("classify",
        lambda s: "bypass" if s["route_name"] == "generic" else "retrieve")
    wf.add_edge("retrieve", END)
    wf.add_edge("bypass",   END)
    return wf.compile()


# ── F3: Streaming with thinking animation + ESC interrupt ────────────────────
def stream_response(llm, prompt: str) -> str:
    import threading
    import select
    import tty
    import termios

    chunks       = []
    done_event   = threading.Event()
    cancel_event = threading.Event()

    def collect():
        try:
            for chunk in llm.stream(prompt):
                if cancel_event.is_set(): break
                chunks.append(chunk)
        finally:
            done_event.set()

    t = threading.Thread(target=collect, daemon=True)
    t.start()

    cancelled = False
    frame_idx = 0
    fd        = sys.stdin.fileno() if sys.stdin.isatty() else None
    old_term  = termios.tcgetattr(fd) if fd is not None else None
    if fd is not None:
        tty.setcbreak(fd)

    try:
        with Live(Text.from_markup(THINK_FRAMES[0]),
                  refresh_per_second=4, transient=True) as live:
            while not done_event.is_set():
                if fd is not None and select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch == "\x1b":
                        cancel_event.set()
                        cancelled = True
                        break
                frame_idx = (frame_idx + 1) % len(THINK_FRAMES)
                live.update(Text.from_markup(THINK_FRAMES[frame_idx]))
                time.sleep(0.25)
    finally:
        if fd is not None and old_term is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_term)

    if cancelled:
        done_event.wait(timeout=2.0)

    ai_response = "".join(chunks)

    console.print()
    console.print(f"  [bold {PRI}]Nexus ›[/bold {PRI}]")
    console.print()
    if ai_response.strip():
        # F3: monokai theme gives proper syntax highlighting on all code blocks + diffs
        console.print(Markdown(ai_response, code_theme="monokai"))
        if cancelled:
            console.print(f"  [dim]↩  interrupted[/dim]")
    else:
        if cancelled:
            console.print(f"  [dim]↩  Cancelled before response started.[/dim]")
    console.print()
    return ai_response


# ── F10: Agent mode ───────────────────────────────────────────────────────────
def run_agent_loop(intent: str, session, llm, root_dir: str) -> str:
    console.print(f"\n  [{ACC}]Agent mode[/{ACC}]  [dim]→  {intent}[/dim]\n")

    # Step 1: scout which files to read
    scout_prompt = (
        "You are a coding agent. Given this task, list the files you need to read. "
        "Output ONLY a JSON array of relative file paths. Example: [\"app/cli.py\", \"ingest.py\"]\n\n"
        f"Task: {intent}\n\n"
        f"Workspace:\n{generate_tree_map(root_dir)}\n\n"
        "Files needed (JSON array only):"
    )
    with Live(Spinner("dots", style=PRI, text=f"  [{PRI}]Scouting files...[/{PRI}]"),
              refresh_per_second=12, transient=True):
        raw_files = llm.invoke(scout_prompt)

    file_matches = re.findall(r'"([^"]+\.\w+)"', raw_files)

    injected    = ""
    found_files = []
    for rel in dict.fromkeys(file_matches[:6]):
        fp = os.path.join(root_dir, rel) if not os.path.isabs(rel) else rel
        if not os.path.exists(fp):
            fp = find_custom_file(os.path.basename(rel), root_dir) or ""
        if fp and os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f: code = f.read()
                injected += f"\n--- File: {fp} ---\n{code}\n"
                found_files.append(fp)
            except Exception: pass

    if not found_files:
        console.print("  [dim]No relevant files found. Running as a regular query.[/dim]\n")
        return ""

    console.print(
        f"  [dim]Reading:[/dim] "
        + ", ".join(f"[white]{os.path.basename(f)}[/white]" for f in found_files)
        + "\n"
    )

    # Step 2: generate the edits
    edit_prompt = (
        "You are a coding agent. Complete the following task.\n"
        "For each file that needs to change, output exactly:\n\n"
        "FILE: <relative/path.py>\n"
        "```<language>\n"
        "<complete new file content — no truncation, no placeholders>\n"
        "```\n\n"
        "Only include files that actually change.\n\n"
        f"Task: {intent}\n\n"
        f"=== CURRENT FILES ===\n{injected}\n====================\n"
    )

    ai_response = stream_response(llm, edit_prompt)

    # Step 3: parse proposed writes
    file_blocks = re.findall(r'FILE:\s*([^\n]+)\n```\w*\n(.*?)```', ai_response, re.DOTALL)

    if not file_blocks:
        console.print("  [dim]No file writes in response — shown above.[/dim]\n")
        return ai_response

    console.print(f"\n  [{PRI}]Agent proposes {len(file_blocks)} write(s):[/{PRI}]\n")
    for path, content in file_blocks:
        console.print(f"  [{YOU}]→[/{YOU}] [white]{path.strip()}[/white]  "
                      f"[dim]({len(content.strip())} chars)[/dim]")

    console.print()
    confirm = console.input(
        f"[dim][[/dim] [{PRI}]Apply?[/{PRI}]  "
        f"[{YOU}]y[/{YOU}][dim]es  [/dim]"
        f"[{YOU}]d[/{YOU}][dim]iff  [/dim]"
        f"[{YOU}]n[/{YOU}][dim]o  ›  [/dim]"
    ).strip().lower()

    if confirm == "d":
        for path, new_content in file_blocks:
            path = path.strip()
            full = os.path.join(root_dir, path) if not os.path.isabs(path) else path
            old_content = ""
            if os.path.exists(full):
                with open(full) as f: old_content = f.read()
            diff_lines = list(difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.strip().splitlines(keepends=True),
                fromfile=f"a/{path}", tofile=f"b/{path}"
            ))
            diff_text = "".join(diff_lines[:80])
            console.print(Panel(
                Markdown(f"```diff\n{diff_text or '(no diff)'}\n```", code_theme="monokai"),
                title=f"[bold {PRI}]  Diff: {path}  [/bold {PRI}]",
                border_style=PRI, box=box.HEAVY, padding=(1, 2)
            ))
        confirm = console.input(
            f"[dim][[/dim] [{PRI}]Apply?[/{PRI}]  [{YOU}]y[/{YOU}][dim]/n  ›  [/dim]"
        ).strip().lower()

    if confirm == "y":
        written = []
        for path, new_content in file_blocks:
            path = path.strip()
            full = os.path.join(root_dir, path) if not os.path.isabs(path) else path
            try:
                os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
                with open(full, "w", encoding="utf-8") as f:
                    f.write(new_content.strip() + "\n")
                written.append(path)
            except Exception as e:
                err(f"Failed to write {path}: {e}")
        if written:
            ok(f"Wrote {len(written)} file(s): {', '.join(written)}")
    else:
        console.print("  [dim]Writes cancelled.[/dim]\n")

    return ai_response


# ── /instructions — paginated new-user guide ─────────────────────────────────
def cmd_instructions() -> None:
    """
    Paginated onboarding guide. One section per page, navigate with
    Enter (next) or q (quit). Written for someone opening Nexus for the first time.
    """
    pages = [
        # ── Page 1 ────────────────────────────────────────────────────────────
        (
            "Welcome to Nexus",
            f"""\
[bold {PRI}]What is Nexus?[/bold {PRI}]

  Nexus is a fully [bold]offline[/bold], privacy-first AI assistant that runs entirely
  on your local machine. No data leaves your computer. All AI models are
  stored on your SSD and served by Ollama.

  Think of it like having a senior engineer sitting next to you who:
    ·  Understands your project files and git history
    ·  Reads and reasons over any document you point it to
    ·  Can propose and apply code edits with your approval
    ·  Searches the web when you need live information

[bold {PRI}]How to open Nexus in a specific project[/bold {PRI}]

  [dim]Option 1 — cd to your project first, then launch:[/dim]
  [bold {YOU}]cd /path/to/my/project[/bold {YOU}]
  [bold {YOU}]./start_nexus.sh[/bold {YOU}]

  [dim]Option 2 — pass the path directly:[/dim]
  [bold {YOU}]./start_nexus.sh /path/to/my/project[/bold {YOU}]

  Nexus will show an [bold]Active Project[/bold] banner when it starts, confirming
  which directory and git branch it is reading.\
""",
        ),

        # ── Page 2 ────────────────────────────────────────────────────────────
        (
            "Ollama Setup & Model Connection",
            f"""\
[bold {PRI}]Install Ollama[/bold {PRI}]

  [dim]macOS:[/dim]   [bold {YOU}]brew install ollama[/bold {YOU}]
  [dim]Linux:[/dim]   [bold {YOU}]curl -fsSL https://ollama.com/install.sh | sh[/bold {YOU}]
  [dim]Or download from:[/dim] [bold white]https://ollama.com[/bold white]

[bold {PRI}]Start the server[/bold {PRI}]

  [bold {YOU}]ollama serve[/bold {YOU}]
  [dim]Run this in a separate terminal before launching Nexus.[/dim]
  [dim](start_nexus.sh does this automatically if you use it)[/dim]

[bold {PRI}]Pull models (one-time per model)[/bold {PRI}]

  [bold {YOU}]ollama pull qwen2.5:14b[/bold {YOU}]       [dim]← recommended chat model[/dim]
  [bold {YOU}]ollama pull llama3[/bold {YOU}]             [dim]← fast alternative[/dim]
  [bold {YOU}]ollama pull nomic-embed-text[/bold {YOU}]   [dim]← required for document search[/dim]

[bold {PRI}]Connect to a different Ollama server[/bold {PRI}]

  By default Nexus connects to [bold white]http://localhost:11434[/bold white].
  If you run Ollama on a different port or remote machine, use:

  [bold {YOU}]/connect http://localhost:8080[/bold {YOU}]          [dim]← different port[/dim]
  [bold {YOU}]/connect http://192.168.1.10:11434[/bold {YOU}]     [dim]← remote machine[/dim]

  The connection is validated and saved to config automatically.

[bold {PRI}]Models on an external drive (SSD / NAS)[/bold {PRI}]

  Enter the path in the setup wizard on first launch, or:
  [bold {YOU}]OLLAMA_MODELS=/Volumes/MySSD/Ollama_Models nexus[/bold {YOU}]

[bold {PRI}]Check current config[/bold {PRI}]

  [bold {YOU}]/config[/bold {YOU}]  → shows Ollama server URL, default model, all paths\
""",
        ),

        # ── Page 3 ────────────────────────────────────────────────────────────
        (
            "Having a Conversation",
            f"""\
[bold {PRI}]Just type and press Enter[/bold {PRI}]

  At the [bold {YOU}]You ›[/bold {YOU}] prompt, type your question naturally.
  Nexus will think, then respond with formatted text and syntax-highlighted code.

[bold {PRI}]While Nexus is thinking[/bold {PRI}]

  You will see a [dim]●○○  thinking…[/dim] animation.
  Press [bold]ESC[/bold] at any time to interrupt the response early.
  Whatever was generated so far will be printed, marked [dim]↩ interrupted[/dim].

[bold {PRI}]After every response[/bold {PRI}]

  A one-line prompt appears:
    [bold {YOU}][ c ][/bold {YOU}] [dim]copy[/dim]   [bold {YOU}][ r ][/bold {YOU}] [dim]run code[/dim]   [dim]Enter to continue[/dim]

  [bold {YOU}]c[/bold {YOU}] → copies the full response to your clipboard instantly.
  [bold {YOU}]r[/bold {YOU}] → if the response contained code, shows it and asks: run it? [y/N]
        Output appears in a panel below. Supports Python, Bash, JavaScript.
  [bold]Enter[/bold] → do nothing, continue chatting.

[bold {PRI}]Stopping the agent entirely[/bold {PRI}]

  Type [bold {YOU}]/exit[/bold {YOU}] or press [bold]Ctrl+C[/bold].
  Your session is saved automatically every time you get a response.\
""",
        ),

        # ── Page 4 ────────────────────────────────────────────────────────────
        (
            "Working with Files",
            f"""\
[bold {PRI}]Inject any file into the conversation[/bold {PRI}]

  Nexus reads the file content and gives it to the model as context.
  You can reference files in your question naturally:

  [bold {YOU}]@filename[/bold {YOU}]
    Searches your workspace for that filename.
    [dim]Example:  "Find the bug in @auth.py"[/dim]

  [bold {YOU}]@/absolute/path/to/file.py[/bold {YOU}]
    Reads any file anywhere on your computer.
    [dim]Example:  "Explain @/Users/me/scripts/deploy.sh"[/dim]

  [bold {YOU}]@~/Desktop/notes.md[/bold {YOU}]
    Supports ~ home directory shorthand.

  [bold {YOU}]@/some/directory[/bold {YOU}]
    Injects a tree listing of the entire directory.

[bold {PRI}]Files with spaces in the name[/bold {PRI}]

  Put the path in quotes (single or double):
  [bold {YOU}]'/Users/me/Desktop/my document (2).pdf'[/bold {YOU}]

  PDFs are read automatically — Nexus extracts the text from each page.

[bold {PRI}]Just paste a path[/bold {PRI}]

  If you paste an absolute path [dim](/Users/me/file.py)[/dim] anywhere in your message,
  Nexus auto-detects it and injects the file without needing @.

[bold {PRI}]View a file without injecting[/bold {PRI}]

  [bold {YOU}]/read /path/to/file[/bold {YOU}]   → syntax-highlighted panel, no LLM call
  [bold {YOU}]/ls /path/to/dir[/bold {YOU}]       → directory listing panel\
""",
        ),

        # ── Page 5 ────────────────────────────────────────────────────────────
        (
            "Project & Git Context",
            f"""\
[bold {PRI}]How Nexus understands your project[/bold {PRI}]

  Every time you start a session, Nexus automatically reads:

    [bold {YOU}]Directory tree[/bold {YOU}]   Up to 200 files and folders, injected into every prompt.
    [bold {YOU}]Git branch[/bold {YOU}]       The active branch name.
    [bold {YOU}]Recent commits[/bold {YOU}]   Last 8 commit messages and hashes.
    [bold {YOU}]Uncommitted diff[/bold {YOU}]  Everything you've changed but not committed yet.

  None of this requires any setup. Nexus runs the standard [bold]git[/bold] CLI
  commands on whatever directory you launched it from.

[bold {PRI}]Nexus does NOT connect to the internet for this[/bold {PRI}]

  Git context is read entirely from the local [dim].git[/dim] folder on disk.
  No GitHub or remote access is involved.

[bold {PRI}]Pin files that are always relevant[/bold {PRI}]

  If you're working on a specific file throughout a session, pin it so it's
  included in every LLM call without typing @filename every time:

  [bold {YOU}]/context add /path/to/important/file.py[/bold {YOU}]  → pin a file
  [bold {YOU}]/context show[/bold {YOU}]                             → see what's pinned
  [bold {YOU}]/context rm 1[/bold {YOU}]                            → unpin item #1
  [bold {YOU}]/context clear[/bold {YOU}]                           → unpin everything\
""",
        ),

        # ── Page 6 ────────────────────────────────────────────────────────────
        (
            "Sessions & History",
            f"""\
[bold {PRI}]Sessions are saved automatically[/bold {PRI}]

  Every conversation is saved as a JSON file in [dim]data/history/[/dim].
  When you start Nexus, you'll see a list of your previous sessions.
  Pick a number to resume any of them — full history is restored.

[bold {PRI}]Naming and organizing[/bold {PRI}]

  New sessions are auto-titled based on your first message.
  You can rename at any time:
  [bold {YOU}]/rename my cool project[/bold {YOU}]

  Tag sessions to find them later:
  [bold {YOU}]/tag python[/bold {YOU}]
  [bold {YOU}]/tag work[/bold {YOU}]

  Search across ALL sessions by keyword or tag:
  [bold {YOU}]/search authentication[/bold {YOU}]
  [bold {YOU}]/search python[/bold {YOU}]

[bold {PRI}]Browsing history[/bold {PRI}]

  [bold {YOU}]/history[/bold {YOU}]   → see all sessions, pick one to read
  [bold {YOU}]/manage[/bold {YOU}]    → delete, rename, or clone a session

[bold {PRI}]Export a session[/bold {PRI}]

  [bold {YOU}]/export[/bold {YOU}]   → saves the full conversation as a Markdown file
            to your Desktop, ready to share or archive.\
""",
        ),

        # ── Page 7 ────────────────────────────────────────────────────────────
        (
            "Web Search & Live Data",
            f"""\
[bold {PRI}]Search the web from inside Nexus[/bold {PRI}]

  Nexus is offline-first, but you can pull in live data when you need it.
  The top 3 results are fetched and injected into the LLM context.

  [dim]As a command:[/dim]
  [bold {YOU}]/web python 3.13 new features[/bold {YOU}]

  [dim]Inline in a message (results become context for your question):[/dim]
  [bold {YOU}]@web:bitcoin price  — what's the latest price?[/bold {YOU}]

  After a [bold {YOU}]/web[/bold {YOU}] search, the results panel is shown.
  Then ask your question and the model will use those results as context.

[bold {PRI}]Requires internet[/bold {PRI}]

  Web search uses DuckDuckGo. Everything else in Nexus works fully offline.
  If the search fails, the error message will explain why.\
""",
        ),

        # ── Page 8 ────────────────────────────────────────────────────────────
        (
            "Power Tools",
            f"""\
[bold {PRI}]/diagnose — root-cause error analysis[/bold {PRI}]

  Paste an error or stack trace. Nexus scans your workspace for the files
  and symbols mentioned, reads them, and produces a structured diagnosis:
  root cause · exact location · concrete fix · explanation.

  [dim]Example:[/dim]
  [bold {YOU}]/diagnose AttributeError: 'NoneType' object has no attribute 'split'[/bold {YOU}]

──────────────────────────────────────────────────

[bold {PRI}]@agent — let Nexus write code for you[/bold {PRI}]

  Agent mode is a read → propose → approve → write loop:
    1.  You describe what you want in plain English.
    2.  Nexus figures out which files to read.
    3.  It proposes complete rewrites of the affected files.
    4.  You review the diff, then type [bold {YOU}]y[/bold {YOU}] to apply or [bold {YOU}]n[/bold {YOU}] to cancel.

  [dim]Example:[/dim]
  [bold {YOU}]@agent add input validation to the login form in auth.py[/bold {YOU}]
  [bold {YOU}]@agent refactor the database module to use connection pooling[/bold {YOU}]

  Files are [bold]never changed without your explicit approval.[/bold]\
""",
        ),

        # ── Page 9 ────────────────────────────────────────────────────────────
        (
            "Plugins & Quick Reference",
            f"""\
[bold {PRI}]Adding plugins[/bold {PRI}]

  Drop a [dim].py[/dim] file into the [dim]plugins/[/dim] folder. It becomes a slash command
  automatically on the next launch. No config needed.

  Your plugin file must export a [dim]PLUGIN[/dim] dict:

  [dim]plugins/myplugin.py:[/dim]
  [bold {YOU}]PLUGIN = {{[/bold {YOU}]
  [bold {YOU}]    "name":        "myplugin",[/bold {YOU}]
  [bold {YOU}]    "description": "What this plugin does",[/bold {YOU}]
  [bold {YOU}]    "run":         lambda arg, ctx: ctx["console"].print(arg)[/bold {YOU}]
  [bold {YOU}]}}[/bold {YOU}]

  [dim]ctx[/dim] gives your plugin access to: [dim]session · llm · console · root_dir[/dim]

──────────────────────────────────────────────────

[bold {PRI}]Quick reference card[/bold {PRI}]

  [bold {YOU}]/help[/bold {YOU}]          Command list
  [bold {YOU}]/instructions[/bold {YOU}]  This guide
  [bold {YOU}]/history[/bold {YOU}]       Browse sessions
  [bold {YOU}]/context[/bold {YOU}]       Pin files
  [bold {YOU}]/web[/bold {YOU}]           Web search
  [bold {YOU}]/export[/bold {YOU}]        Save to Desktop
  [bold {YOU}]/tag[/bold {YOU}]           Label a session
  [bold {YOU}]/search[/bold {YOU}]        Find past sessions
  [bold {YOU}]/diagnose[/bold {YOU}]      Debug an error
  [bold {YOU}]/read[/bold {YOU}]          View a file
  [bold {YOU}]/ls[/bold {YOU}]            Browse a directory
  [bold {YOU}]/rename[/bold {YOU}]        Rename session
  [bold {YOU}]/manage[/bold {YOU}]        Delete / clone sessions
  [bold {YOU}]/pr[/bold {YOU}]            PR documentation generator
  [bold {YOU}]@agent[/bold {YOU}]         File edit mode
  [bold {YOU}]/connect[/bold {YOU}] [dim]<url>[/dim]   Change Ollama server
  [bold {YOU}]/config[/bold {YOU}]        Show active server, model, paths
  [bold {YOU}]/exit[/bold {YOU}]          Save and quit\
""",
        ),
    ]

    total = len(pages)

    for i, (title, body) in enumerate(pages, start=1):
        console.print()
        console.print(Panel(
            body,
            title=(
                f"[bold {PRI}]  {title}  [/bold {PRI}]"
                f"[dim]  ({i}/{total})[/dim]"
            ),
            border_style=PRI,
            box=box.HEAVY,
            padding=(1, 3),
        ))

        if i < total:
            console.print(
                f"  [dim][[/dim] [bold]Enter[/bold] [dim]next page[/dim]"
                f"  [bold {YOU}]q[/bold {YOU}] [dim]quit guide ][/dim]",
                end="",
            )
            sys.stdout.flush()
            ch = read_key()
            console.print()
            if ch.lower() == "q":
                console.print("  [dim]Guide closed.[/dim]\n")
                return
        else:
            console.print(
                f"\n  [dim {ACC}]End of guide. Type /help for the command list.[/dim {ACC}]\n"
            )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    load_plugins()
    render_splash(animated=True)

    langgraph_app = None
    quit_entirely = False

    while not quit_entirely:

        # ── Session picker ────────────────────────────────────────────────────
        session_id = render_session_menu()

        if session_id == "__pr__":
            handle_pr_generator()
            console.print(f"  [bold {PRI}]Back to session picker:[/bold {PRI}]")
            console.print(); continue

        session = ChatSession(session_id=session_id)
        session.load()

        # ── Model selection ───────────────────────────────────────────────────
        model_name    = render_model_selector()
        session.model = model_name
        llm = OllamaLLM(model=model_name, num_thread=8, num_ctx=32768, keep_alive="30m")

        # ── Resume banner ─────────────────────────────────────────────────────
        if session.history:
            console.print()
            console.print(Panel(
                f"[dim]Resuming [bold white]{session.session_id.replace('_',' ')}[/bold white]  "
                f"·  {len(session.history)//2} previous exchange(s)  "
                f"·  type [bold {YOU}]/history[/bold {YOU}] to browse all sessions[/dim]",
                border_style=f"dim {PRI}", box=box.HEAVY, padding=(0, 2),
            ))

        # ── F7: Git context (once per session) ────────────────────────────────
        project_dir              = os.getcwd()
        git_branch, git_context  = get_git_context(project_dir)
        cwd_tree_map             = generate_tree_map(project_dir)

        # Project context banner — always visible
        project_name = os.path.basename(project_dir)
        ctx_parts    = [f"[bold white]{project_name}[/bold white]  [dim]{project_dir}[/dim]"]
        if git_branch:
            ctx_parts.append(f"[dim]branch[/dim] [bold {ACC}]{git_branch}[/bold {ACC}]")
        ctx_parts.append(f"[dim]{cwd_tree_map.count(chr(10))} files mapped[/dim]")

        console.print()
        console.print(Panel(
            "  " + "  ·  ".join(ctx_parts),
            title=f"[bold {PRI}]  Active Project  [/bold {PRI}]",
            border_style=f"dim {PRI}", box=box.HEAVY, padding=(0, 2),
        ))

        render_status_bar(session, git_branch)
        console.print(f"  [dim]Type [{YOU}]/help[/{YOU}] for all commands  ·  "
                      f"[@agent intent] for file edits  ·  [ESC] to interrupt[/dim]")
        console.print()

        # ── Chat loop ─────────────────────────────────────────────────────────
        switch_session = False
        while not switch_session:
            try:
                user_input = console.input(
                    f"[dim][[/dim] [bold {YOU}]You[/bold {YOU}] [{PRI}]›[/{PRI}]  "
                )

                # ── Slash commands ────────────────────────────────────────────
                if user_input.startswith("/"):
                    raw   = user_input[1:].strip()
                    parts = raw.split(maxsplit=1)
                    cmd   = parts[0].lower() if parts else ""
                    arg   = parts[1].strip() if len(parts) > 1 else ""

                    if cmd in ("exit", "quit"):
                        session.save()
                        console.print(f"\n  [dim]Session saved. Goodbye.[/dim]\n")
                        quit_entirely = switch_session = True

                    elif cmd == "help":
                        console.print(Panel(build_chat_help(),
                            title=f"[bold {PRI}]  Nexus Help  [/bold {PRI}]",
                            border_style=PRI, box=box.HEAVY, padding=(1, 3)))

                    elif cmd == "history":
                        cmd_history()

                    elif cmd == "clear":
                        render_splash(animated=False)
                        render_status_bar(session, git_branch)

                    elif cmd == "rename":
                        name = arg or console.input(
                            f"[dim][[/dim] [{PRI}]New name[/{PRI}] [{PRI}]›[/{PRI}]  ").strip()
                        name = name.replace(" ", "_").lower()
                        if name:
                            old = session.session_id
                            session.rename(name)
                            ok(f"Renamed: {old.replace('_',' ')}  →  {name.replace('_',' ')}")
                        else:
                            err("Name cannot be empty.")

                    elif cmd == "sessions":
                        session.save(); console.print(); switch_session = True

                    elif cmd == "manage":
                        sessions = list_sessions()
                        result = _manage_sessions_inline(sessions)
                        if result not in (None, "__refresh__"):
                            session.save()
                            session = ChatSession(session_id=result)
                            session.load(); session.model = model_name
                            render_status_bar(session, git_branch)

                    elif cmd == "pr":
                        handle_pr_generator()

                    elif cmd == "model":
                        console.print(f"\n  [dim]Active model:[/dim] [bold {ACC}]{session.model}[/bold {ACC}]\n")

                    elif cmd == "connect":
                        host = (arg.strip().rstrip("/") or "").replace("http://","").replace("https://","")
                        if not host:
                            console.print(f"\n  [dim]Usage: /connect http://host:11434[/dim]\n")
                        else:
                            full = f"http://{host}"
                            try:
                                import requests as _req
                                r = _req.get(f"{full}/api/tags", timeout=5)
                                if r.status_code == 200:
                                    os.environ["OLLAMA_HOST"] = full
                                    llm = OllamaLLM(model=model_name, num_thread=8, num_ctx=32768, keep_alive="30m")
                                    ok(f"Connected to {full}  ·  model: {model_name}")
                                else:
                                    err(f"Ollama at {full} returned HTTP {r.status_code}")
                            except Exception as e:
                                err(f"Cannot reach {full}: {e}")

                    elif cmd == "read":
                        if not arg:
                            err("Usage: /read <path>")
                        else:
                            rp = os.path.expanduser(arg.strip())
                            if not os.path.exists(rp):
                                err(f"Path not found: {rp}")
                            elif os.path.isdir(rp):
                                tree = generate_tree_map(rp)
                                console.print(Panel(
                                    f"[dim]{tree}[/dim]",
                                    title=f"[bold {PRI}]  {rp}  [/bold {PRI}]",
                                    subtitle=f"[dim]  directory tree  [/dim]",
                                    border_style=PRI, box=box.HEAVY, padding=(1, 2)
                                ))
                            else:
                                try:
                                    with open(rp, "r", encoding="utf-8") as f:
                                        content = f.read()
                                    ext  = os.path.splitext(rp)[1].lstrip(".")
                                    body = f"```{ext}\n{content}\n```" if ext else content
                                    console.print(Panel(
                                        Markdown(body, code_theme="monokai"),
                                        title=f"[bold {PRI}]  {os.path.basename(rp)}  [/bold {PRI}]",
                                        subtitle=f"[dim]  {rp}  [/dim]",
                                        border_style=PRI, box=box.HEAVY, padding=(1, 2)
                                    ))
                                except Exception as e:
                                    err(f"Cannot read {rp}: {e}")

                    elif cmd == "ls":
                        target = os.path.expanduser(arg.strip()) if arg else os.getcwd()
                        if not os.path.exists(target):
                            err(f"Path not found: {target}")
                        elif not os.path.isdir(target):
                            err(f"Not a directory: {target}")
                        else:
                            try:
                                entries = sorted(os.listdir(target))
                                lines   = []
                                for e in entries:
                                    full = os.path.join(target, e)
                                    if os.path.isdir(full):
                                        lines.append(f"[bold {YOU}]  {e}/[/bold {YOU}]")
                                    elif not e.startswith("."):
                                        lines.append(f"  [white]{e}[/white]")
                                    else:
                                        lines.append(f"  [dim]{e}[/dim]")
                                console.print(Panel(
                                    "\n".join(lines) or "[dim]  (empty)[/dim]",
                                    title=f"[bold {PRI}]  {target}  [/bold {PRI}]",
                                    subtitle=f"[dim]  {len(entries)} item(s)  [/dim]",
                                    border_style=PRI, box=box.HEAVY, padding=(1, 2)
                                ))
                            except PermissionError:
                                err(f"Permission denied: {target}")

                    elif cmd == "context":
                        cmd_context(session, arg)

                    elif cmd == "web":
                        if not arg:
                            err("Usage: /web <search query>")
                        else:
                            with Live(Spinner("dots", style=PRI,
                                             text=f"  [{PRI}]Searching the web...[/{PRI}]"),
                                      refresh_per_second=12, transient=True):
                                result = web_search(arg)
                            console.print(Panel(
                                Markdown(result),
                                title=f"[bold {PRI}]  Web: {arg[:60]}  [/bold {PRI}]",
                                border_style=PRI, box=box.HEAVY, padding=(1, 2)
                            ))
                            # Also inject into next message if user wants
                            console.print(
                                f"  [dim]Results shown above. Ask a question to use them as context.[/dim]\n"
                            )

                    elif cmd == "export":
                        cmd_export(session)

                    elif cmd == "tag":
                        cmd_tag(session, arg)

                    elif cmd == "search":
                        cmd_search(arg)

                    elif cmd == "diagnose":
                        cmd_diagnose(arg, session, llm, ROOT_DIR)

                    elif cmd in ("instructions", "guide"):
                        cmd_instructions()

                    elif cmd in _loaded_plugins:
                        plugin = _loaded_plugins[cmd]
                        try:
                            plugin["run"](arg, {
                                "session": session, "llm": llm,
                                "console": console, "root_dir": ROOT_DIR,
                            })
                        except Exception as e:
                            err(f"Plugin /{cmd} error: {e}")

                    else:
                        err(f"Unknown command '/{cmd}'. Type /help to see all commands.")

                    continue

                # Legacy bare exit
                if user_input.lower() in ("exit", "quit"):
                    session.save()
                    console.print(f"\n  [dim]Session saved. Goodbye.[/dim]\n")
                    quit_entirely = switch_session = True
                    continue

                if not user_input.strip():
                    continue

                # ── F10: Agent mode ───────────────────────────────────────────
                if user_input.strip().startswith("@agent"):
                    intent = user_input.strip()[6:].strip()
                    if not intent:
                        err("Usage: @agent <intent — what you want done>")
                        continue
                    ai_response = run_agent_loop(intent, session, llm, ROOT_DIR)
                    if ai_response:
                        session.history.append(HumanMessage(content=user_input))
                        session.history.append(AIMessage(content=ai_response))
                        session.save()
                        post_response_prompt(ai_response)
                    continue

                # Auto-title new UUID sessions
                is_uuid = len(session.session_id) == 36 and "-" in session.session_id
                if not session.history and is_uuid:
                    with Live(Spinner("dots", style=PRI,
                                      text=f"  [{PRI}]Generating session title...[/{PRI}]"),
                              refresh_per_second=12, transient=True):
                        try:
                            raw = llm.invoke(
                                "Summarize the core topic of this prompt in 3–7 words. "
                                "Return ONLY the words separated by underscores, no punctuation: "
                                + user_input
                            )
                            title = (raw.strip().replace(" ","_").replace("'","").replace('"',"")
                                        .replace(".","").replace("/","").replace("\n","")
                                        .lower()[:60].strip("_"))
                            if title: session.rename(title)
                        except Exception: pass

                # ── @token injection ──────────────────────────────────────────
                injected       = ""
                injected_paths = set()
                web_context    = ""

                # F2: @web:query inline injection
                for web_q in re.findall(r'@web:([^\s@]+)', user_input):
                    with Live(Spinner("dots", style=PRI,
                                     text=f"  [{PRI}]Web search: {web_q[:40]}...[/{PRI}]"),
                              refresh_per_second=12, transient=True):
                        web_context += web_search(web_q) + "\n\n"
                    console.print(f"  [dim {ACC}]✓ Web results injected: {web_q}[/dim {ACC}]")

                # File injection: @/path, @~/path, @filename
                tokens = re.findall(r'@([\w./~\-]+)', user_input)
                inp    = user_input

                for token in dict.fromkeys(tokens):
                    if token.startswith("web:"): continue
                    inp      = inp.replace(f"@{token}", f"`{token}`")
                    expanded = os.path.expanduser(token)

                    if token.startswith("/") or token.startswith("~"):
                        resolved = expanded if os.path.exists(expanded) else None
                    else:
                        resolved = find_custom_file(token, os.getcwd())

                    if resolved:
                        injected_paths.add(os.path.realpath(resolved))
                        if os.path.isdir(resolved):
                            tree = generate_tree_map(resolved)
                            injected += f"\n--- Directory Tree: {resolved} ---\n{tree}\n"
                            console.print(f"  [dim {ACC}]✓ Injected tree: {token}[/dim {ACC}]")
                        else:
                            content = read_any_path(resolved)
                            injected += f"\n--- Source: {resolved} ---\n{content}\n"
                            console.print(f"  [dim {ACC}]✓ Injected: {os.path.basename(resolved)}[/dim {ACC}]")
                    else:
                        console.print(f"  [dim]⚠  Not found: {token}[/dim]")

                # Detect quoted paths — handles spaces, parentheses, any filename chars
                # e.g.  '/Users/me/Desktop/document_pdf (2).pdf'  or  "/path/to file.txt"
                for m in re.finditer(r"""['"](/[^'"]+)['"]""", user_input):
                    candidate = m.group(1).strip()
                    if not os.path.exists(candidate): continue
                    real = os.path.realpath(candidate)
                    if real in injected_paths: continue
                    injected_paths.add(real)
                    if os.path.isdir(candidate):
                        tree = generate_tree_map(candidate)
                        injected += f"\n--- Directory Tree: {candidate} ---\n{tree}\n"
                        console.print(f"  [dim {ACC}]✓ Injected tree: {os.path.basename(candidate)}[/dim {ACC}]")
                    else:
                        content = read_any_path(candidate)
                        injected += f"\n--- Source: {candidate} ---\n{content}\n"
                        console.print(f"  [dim {ACC}]✓ Injected: {os.path.basename(candidate)}[/dim {ACC}]")

                # Auto-detect bare absolute paths (no spaces — use quotes for those)
                for candidate in dict.fromkeys(
                        re.findall(r'(?<![\'"/\w])((?:/[\w.\-()]+){2,})', user_input)):
                    expanded = os.path.expanduser(candidate)
                    if not os.path.exists(expanded): continue
                    real = os.path.realpath(expanded)
                    if real in injected_paths: continue
                    injected_paths.add(real)
                    if os.path.isdir(expanded):
                        tree = generate_tree_map(expanded)
                        injected += f"\n--- Directory Tree: {expanded} ---\n{tree}\n"
                        console.print(f"  [dim {ACC}]✓ Auto-injected tree: {expanded}[/dim {ACC}]")
                    else:
                        content = read_any_path(expanded)
                        injected += f"\n--- Source: {expanded} ---\n{content}\n"
                        console.print(f"  [dim {ACC}]✓ Auto-injected: {os.path.basename(expanded)}[/dim {ACC}]")

                # Detect paths with spaces that end in a known file extension.
                # Catches: /Users/me/Desktop/document name (2).pdf  — space breaks the bare-path regex above.
                for m in re.finditer(
                    r'(/[^\n\'"]*?\.(?:pdf|docx|pptx|xlsx|xls|txt|md|py|js|ts|json|yaml|yml|csv))',
                    user_input, re.IGNORECASE
                ):
                    candidate = m.group(1).strip()
                    if not os.path.exists(candidate): continue
                    real = os.path.realpath(candidate)
                    if real in injected_paths: continue
                    injected_paths.add(real)
                    content = read_any_path(candidate)
                    injected += f"\n--- Source: {candidate} ---\n{content}\n"
                    console.print(f"  [dim {ACC}]✓ Injected: {os.path.basename(candidate)}[/dim {ACC}]")

                # Merge web context into injected
                if web_context:
                    injected = f"\n=== WEB SEARCH RESULTS ===\n{web_context}\n==========================\n" + injected

                session.history.append(HumanMessage(content=inp))

                # Lazy-init LangGraph
                if langgraph_app is None:
                    with Live(Spinner("dots", style=PRI,
                                      text=f"  [{PRI}]Initializing semantic router (once only)...[/{PRI}]"),
                              refresh_per_second=12, transient=True):
                        langgraph_app = build_langgraph(ROOT_DIR)

                # Route
                with Live(Spinner("arc", style=PRI,
                                  text=f"  [{PRI}]Routing query...[/{PRI}]"),
                          refresh_per_second=12, transient=True):
                    fs = langgraph_app.invoke({
                        "question": inp, "route_name": "generic", "context_text": ""})

                route_name   = fs["route_name"]
                context_text = fs.get("context_text", "")

                if route_name != "generic" and context_text:
                    frags = context_text.count("---") // 2
                    console.print(
                        f"  [dim]Route: [bold {ACC}]{route_name.upper()}[/bold {ACC}]  "
                        f"·  {frags} fragment(s) retrieved[/dim]"
                    )

                # Build pinned context
                pinned_ctx = build_pinned_context(session.pinned)

                # Build prompt (F1 pinned, F7 git, existing injected + context)
                sysp   = session.expert_system if route_name != "generic" else session.generic_system
                prompt = extract_prompt(
                    [sysp] + session.history,
                    context_text   = context_text,
                    injected_code  = injected,
                    tree_map       = cwd_tree_map,
                    git_context    = git_context,
                    pinned_context = pinned_ctx,
                )

                ai_response = stream_response(llm, prompt)

                session.history.append(AIMessage(content=ai_response))
                session.save()

                # F4+F6: post-response copy/run prompt
                post_response_prompt(ai_response)

            except KeyboardInterrupt:
                console.print(f"\n  [dim]Interrupted. Saving...[/dim]\n")
                session.save(); quit_entirely = switch_session = True

            except Exception as e:
                err(str(e))

if __name__ == "__main__":
    main()
