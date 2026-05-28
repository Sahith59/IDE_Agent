"""
Central config and path management for Nexus.
All persistent data lives in ~/.nexus/
"""
import os
import json
import subprocess

NEXUS_HOME          = os.path.expanduser("~/.nexus")
CONFIG_FILE         = os.path.join(NEXUS_HOME, "config.json")
HISTORY_DIR         = os.path.join(NEXUS_HOME, "history")
USER_PLUGINS_DIR    = os.path.join(NEXUS_HOME, "plugins")
CHROMA_DB_DIR       = os.path.join(NEXUS_HOME, "chroma_db")
BM25_INDEX_PATH     = os.path.join(NEXUS_HOME, "bm25_index.pkl")
HF_CACHE_DIR        = os.path.join(NEXUS_HOME, "hf_cache")
RAW_DOCS_DIR        = os.path.join(NEXUS_HOME, "raw_docs")
BUILTIN_PLUGINS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugins")

for _d in (NEXUS_HOME, HISTORY_DIR, USER_PLUGINS_DIR, RAW_DOCS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    os.makedirs(NEXUS_HOME, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def is_setup_done() -> bool:
    cfg = get_config()
    return bool(cfg.get("setup_complete") and cfg.get("default_model"))


def get_default_model() -> str:
    return get_config().get("default_model", "qwen2.5:14b")


def get_ollama_models_path() -> str | None:
    env_path = os.environ.get("OLLAMA_MODELS")
    if env_path and os.path.isdir(env_path):
        return env_path
    cfg_path = get_config().get("ollama_models_path", "")
    if cfg_path and os.path.isdir(cfg_path):
        return cfg_path
    return None


def discover_models(models_path: str | None = None) -> list[tuple[str, str]]:
    """
    Returns list of (model_name, size_str) tuples.
    Tries `ollama list` first; falls back to scanning the manifests directory.
    """
    models: list[tuple[str, str]] = []

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=8
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n")[1:]:
                parts = line.split()
                if not parts:
                    continue
                name = parts[0]
                size = " ".join(parts[3:5]) if len(parts) >= 5 else ""
                models.append((name, size))
    except Exception:
        pass

    if models:
        return models

    if not models_path:
        models_path = get_ollama_models_path()
    if models_path:
        manifests = os.path.join(models_path, "manifests", "registry.ollama.ai", "library")
        if os.path.isdir(manifests):
            for model_name in sorted(os.listdir(manifests)):
                model_dir = os.path.join(manifests, model_name)
                if os.path.isdir(model_dir):
                    for tag in sorted(os.listdir(model_dir)):
                        label = model_name if tag == "latest" else f"{model_name}:{tag}"
                        models.append((label, ""))

    return models


def apply_env() -> None:
    """Apply config-based env vars before importing heavy libraries."""
    cfg = get_config()
    models_path = cfg.get("ollama_models_path") or os.environ.get("OLLAMA_MODELS", "")
    if models_path:
        os.environ["OLLAMA_MODELS"] = models_path
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
