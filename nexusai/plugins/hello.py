"""
Sample Nexus plugin — /hello
Demonstrates the plugin API: receives (arg, ctx) where ctx has session, llm, console, root_dir.
"""
from rich.panel import Panel
from rich import box

PRI = "#FF4500"
YOU = "#FFB347"

def run(arg, ctx):
    c       = ctx["console"]
    session = ctx["session"]
    greeting = arg.strip() if arg else "World"
    c.print(Panel(
        f"  [bold {YOU}]Hello, {greeting}![/bold {YOU}]\n\n"
        f"  [dim]Session:[/dim] [white]{session.session_id.replace('_', ' ')}[/white]\n"
        f"  [dim]Model:[/dim]   [white]{session.model}[/white]\n\n"
        f"  [dim]This is a sample plugin. Drop .py files in ~/.nexus/plugins/\n"
        f"  and they automatically become slash commands.[/dim]",
        title=f"[bold {PRI}]  /hello plugin  [/bold {PRI}]",
        border_style=PRI, box=box.HEAVY, padding=(1, 2)
    ))

PLUGIN = {
    "name":        "hello",
    "description": "Sample plugin — prints a greeting. Usage: /hello <name>",
    "run":         run,
}
