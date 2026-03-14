import os
import json
import uuid
import glob
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, IntPrompt
from rich.markdown import Markdown
from rich.table import Table
from rich.live import Live

console = Console()

# Environment paths setup
APP_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.abspath(os.path.join(APP_DIR, '..', 'data', 'history'))
os.makedirs(HISTORY_DIR, exist_ok=True)

# Important: Force HuggingFace to download the 1.3GB mixedbread model onto the external SSD, not the internal Mac drive
os.environ["HF_HOME"] = "/Volumes/Sahith_SSD/Ollama_Models/HF_Cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

class ChatSession:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.filepath = os.path.join(HISTORY_DIR, f"session_{self.session_id}.json")
        self.history = []
        
        self.expert_system_prompt = SystemMessage(
            content="""You are an IDE Product Expert Agent. You are a fully local, privacy-first AI agent that answers questions based ONLY on the provided Background Context retrieved from the user's knowledge base.
If the answer cannot be deduced from the core context provided to you, clearly state "I do not know based on the provided documents." DO NOT hallucinate, guess, or provide information outside the given context.
"""
        )
        
        self.generic_system_prompt = SystemMessage(
            content="""You are a brilliant, world-class Senior Software Engineer and AI Assistant. You are incredibly helpful, write highly optimized code, and answer general questions with stunning accuracy.
Feel free to completely utilize your pre-trained knowledge to answer the user's questions, write scripts, brainstorm algorithms, and chat.
"""
        )

    def load(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
                    for msg in data:
                        if msg['type'] == 'human':
                            self.history.append(HumanMessage(content=msg['content']))
                        elif msg['type'] == 'ai':
                            self.history.append(AIMessage(content=msg['content']))
                console.print(f"[dim green]Loaded session '{self.session_id}' ({len(self.history)} messages)[/dim green]")
            except json.JSONDecodeError:
                console.print("[bold red]Error loading session. Starting fresh.[/bold red]")
                self.history = []
        else:
            console.print(f"[dim cyan]Started new session: {self.session_id}[/dim cyan]")

    def save(self):
        data = []
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                data.append({"type": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                data.append({"type": "ai", "content": msg.content})
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def print_history(self):
        if not self.history:
            return
        console.print("\n[dim]--- Previous Conversation ---[/dim]")
        for msg in self.history:
            if isinstance(msg, HumanMessage):
                console.print(f"[bold orange3]You:[/bold orange3] {msg.content}")
            else:
                md = Markdown(msg.content)
                console.print(f"[bold cyan]Nexus:[/bold cyan]")
                console.print(md)
        console.print("[dim]-----------------------------[/dim]\n")

def list_sessions():
    files = glob.glob(os.path.join(HISTORY_DIR, "session_*.json"))
    sessions = []
    for f in files:
        basename = os.path.basename(f)
        session_id = basename.replace("session_", "").replace(".json", "")
        # Get brief info
        try:
            with open(f, 'r') as file_obj:
                data = json.load(file_obj)
                length = len(data)
                sessions.append((session_id, length))
        except:
            pass
    return sessions

def extract_prompt(messages, context_text=""):
    prompt = messages[0].content + "\n\n"
    if context_text:
        prompt += f"=== RELEVANT BACKGROUND CONTEXT ===\n{context_text}\n===================================\n\n"
    for msg in messages[1:]:
        if isinstance(msg, HumanMessage):
            prompt += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            prompt += f"Agent: {msg.content}\n"
    prompt += "Agent: "
    return prompt

def render_header():
    title = Text("Nexus IDE Expert Agent", style="bold cyan")
    subtitle = Text("v1.0.0 | Local Offline Mode", style="dim")
    
    welcome_text = Text.assemble(
        ("Welcome to your private AI workspace.\n", "bold white"),
        ("Features: Local inference, chat history, markdown rendering.", "dim")
    )

    panel = Panel(
        welcome_text,
        title=title,
        subtitle=subtitle,
        border_style="orange3",
        padding=(1, 2)
    )
    console.print(panel)

def session_menu():
    while True:
        sessions = list_sessions()
        if not sessions:
            return None

        table = Table(title="Recent Sessions", title_style="bold orange3", border_style="dim", show_header=True, header_style="bold cyan")
        table.add_column("ID", justify="center")
        table.add_column("Session Name", style="white")
        table.add_column("Messages", justify="right")

        for idx, (sid, length) in enumerate(sessions):
            table.add_row(str(idx), sid, str(length))

        console.print(table)
        
        console.print("[dim]Options: \\[number] = Load | d \\[number] = Delete | r \\[number] \\[new_id] = Rename | b \\[number] \\[new_id] = Branch/Clone | p = PR Generator | \\[Enter] = New Session[/dim]")
        choice = console.input("[bold orange3]Select Option:[/bold orange3] ").strip()
        
        if not choice:
            return None
            
        parts = choice.split()
        if choice.lower() == 'p':
            import sys
            root_dir = os.path.abspath(os.path.join(APP_DIR, '..'))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            try:
                from pr_generator.generator import run
                data_dir = os.path.join(root_dir, 'data')
                run(data_dir)
                input("\nPress Enter to return to menu...")
                os.system('cls' if os.name == 'nt' else 'clear')
                render_header()
            except ImportError as e:
                console.print(f"[bold red]Failed to load PR generator backend: {e}[/bold red]")
            continue
            
        elif len(parts) == 1 and parts[0].isdigit():
            idx = int(parts[0])
            if idx < len(sessions):
                return sessions[idx][0]
                
        elif len(parts) == 2 and parts[0].lower() == 'd' and parts[1].isdigit():
            idx = int(parts[1])
            if idx < len(sessions):
                sid = sessions[idx][0]
                filepath = os.path.join(HISTORY_DIR, f"session_{sid}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                    console.print(f"[green]Removed session:[/green] {sid}")
                    
        elif len(parts) >= 3 and parts[0].lower() == 'r' and parts[1].isdigit():
            idx = int(parts[1])
            new_id = "_".join(parts[2:])
            if idx < len(sessions):
                old_sid = sessions[idx][0]
                old_path = os.path.join(HISTORY_DIR, f"session_{old_sid}.json")
                new_path = os.path.join(HISTORY_DIR, f"session_{new_id}.json")
                if os.path.exists(old_path):
                    os.rename(old_path, new_path)
                    console.print(f"[green]Renamed session[/green] '{old_sid}' -> '{new_id}'")
                    
        elif len(parts) >= 3 and parts[0].lower() == 'b' and parts[1].isdigit():
            idx = int(parts[1])
            new_id = "_".join(parts[2:])
            if idx < len(sessions):
                old_sid = sessions[idx][0]
                old_path = os.path.join(HISTORY_DIR, f"session_{old_sid}.json")
                new_path = os.path.join(HISTORY_DIR, f"session_{new_id}.json")
                if os.path.exists(old_path):
                    import shutil
                    shutil.copy2(old_path, new_path)
                    console.print(f"[green]Branched session[/green] '{old_sid}' -> '{new_id}'")
        else:
            console.print("[red]Invalid choice, try again.[/red]")

def main():
    # Clear console (cross-platform)
    os.system('cls' if os.name == 'nt' else 'clear')
    render_header()

    session_id = session_menu()
    session = ChatSession(session_id=session_id)
    session.load()
    session.print_history()

    AVAILABLE_MODELS = ["llama3", "qwen2.5:14b", "nomic-embed-text"]
    console.print("\n[bold orange3]Available Models:[/bold orange3]")
    for idx, m in enumerate(AVAILABLE_MODELS):
        if "nomic" in m:
             console.print(f"[{idx}] [dim]{m} (Not for chat)[/dim]")
        else:
            console.print(f"\\[{idx}] [bold cyan]{m}[/bold cyan]")
        
    model_choice = console.input("[bold orange3]Select a model (Enter a number, or press Enter for default 'llama3'):[/bold orange3] ")
    if model_choice.strip().isdigit() and int(model_choice) < len(AVAILABLE_MODELS):
        MODEL_NAME = AVAILABLE_MODELS[int(model_choice)]
    else:
        MODEL_NAME = "llama3"

    llm = OllamaLLM(
        model=MODEL_NAME,
        num_thread=8,
        keep_alive="30m",
    )
    
    console.print(f"  [reverse green] Ready [/reverse green] Chatting with [bold white]{MODEL_NAME}[/bold white]\n")

    while True:
        try:
            # Using prompt toolkit or just rich input
            user_input = console.input("\n[bold orange3]You ❯[/bold orange3] ")
            
            if user_input.lower() in ['exit', 'quit']:
                console.print("\n[dim]Saving session and exiting...[/dim]")
                session.save()
                break
            if not user_input.strip():
                continue
                
            # If this is the very first message of a new session (UUID), generate a title using LLM
            if len(session.history) == 0 and len(session.session_id) == 36 and '-' in session.session_id:
                console.print("[dim]Auto-generating session title...[/dim]")
                try:
                    title_prompt = f"Summarize the core topic of this prompt in a short, meaningful phrase (3 to 7 words). Return ONLY the words, separated by underscores. No quotes, no spaces, no intro, no punctuation: {user_input}"
                    title_response = llm.invoke(title_prompt)
                    new_title = title_response.strip().replace(" ", "_").replace("'", "").replace('"', '').replace(".", "").replace("/", "").replace("\n", "").lower()[:60]
                    new_title = new_title.strip("_")
                    if new_title:
                        old_filepath = session.filepath
                        session.session_id = new_title
                        session.filepath = os.path.join(HISTORY_DIR, f"session_{session.session_id}.json")
                        if os.path.exists(old_filepath):
                            os.rename(old_filepath, session.filepath)
                except Exception as e:
                    pass # Fail silently and keep the UUID if titling fails

            session.history.append(HumanMessage(content=user_input))
            
            # Semantic Routing - Fast Intent Classification
            # We use a lightweight HuggingFace encoder to vectorize the intent in milliseconds
            console.print("[dim]Analyzing query intent...[/dim]")
            
            # Lazy initialize the router
            if 'router' not in locals():
                from semantic_router import Route
                from semantic_router import SemanticRouter
                from semantic_router.encoders import HuggingFaceEncoder
                
                console.print("[dim]Loading Enterprise Router (mxbai-embed-large-v1)...[/dim]")
                encoder = HuggingFaceEncoder(name="mixedbread-ai/mxbai-embed-large-v1")
                
                # Define intent clusters
                generic_route = Route(
                    name="generic",
                    utterances=[
                        "hello", "hi", "how are you", "write a python script", 
                        "create binary search", "refactor this code", "fix this error",
                        "what is python", "debug this loop", "who are you"
                    ]
                )
                
                expert_route = Route(
                    name="expert",
                    utterances=[
                        "how does the IDE billing work", "explain the flowchart", 
                        "what are the business rules", "how do I configure the server",
                        "read the documentation", "what did the architect say about",
                        "search the manual for errors"
                    ]
                )
                router = SemanticRouter(encoder=encoder, routes=[generic_route, expert_route], auto_sync="local")
                
            route = router(user_input)
            
            # RAG Retrieval - ONLY triggers if it's NOT a generic query
            context_text = ""
            if getattr(route, 'name', 'expert') != 'generic':
                chroma_dir = os.path.join(APP_DIR, '..', 'data', 'chroma_db')
                if os.path.exists(chroma_dir):
                    try:
                        from langchain_chroma import Chroma
                        from langchain_ollama import OllamaEmbeddings
                        
                        console.print("[dim]Expert Route Detected: Searching knowledge base...[/dim]")
                        embeddings = OllamaEmbeddings(model="nomic-embed-text")
                        vector_db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
                        
                        results = vector_db.similarity_search(user_input, k=4)
                        if results:
                            context_text = "\n\n".join([f"--- Snippet ---\n{doc.page_content}" for doc in results])
                            console.print(f"[dim green]✓ Found {len(results)} relevant document snippets.[/dim green]")
                    except Exception as e:
                        console.print(f"[dim red]Warning: RAG retrieval failed: {e}[/dim red]")
            else:
                console.print("[dim]Generic Route Detected: Bypassing Database...[/dim]")
            
            # Select the correct System Prompt based on the Route
            active_system_prompt = session.expert_system_prompt if getattr(route, 'name', 'expert') != 'generic' else session.generic_system_prompt
            all_messages = [active_system_prompt] + session.history
            raw_prompt = extract_prompt(all_messages, context_text)

            console.print("[bold cyan]Nexus ❯[/bold cyan]")
            
            ai_response = ""
            # Stream the raw output first, then render it as markdown after it's fully generated
            # for a smoother typing effect without breaking markdown tables mid-stream
            with Live(Text("...", style="blink dim"), refresh_per_second=15, transient=True) as live:
                for chunk in llm.stream(raw_prompt):
                    ai_response += chunk
                    live.update(Text(ai_response))
            
            # Print Final beautifully rendered markdown response
            console.print(Markdown(ai_response))
            
            session.history.append(AIMessage(content=ai_response))
            session.save()

        except KeyboardInterrupt:
            console.print("\n[red]Session interrupted. Saving...[/red]")
            session.save()
            break
        except Exception as e:
            console.print(f"\n[bold red]Error during inference:[/bold red] {e}")

if __name__ == "__main__":
    main()
