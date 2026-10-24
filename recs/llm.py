"""Ollama over HTTP with the standard library, started only when needed and stopped with its children."""
import json
import os
import subprocess
import time
import urllib.request
from contextlib import contextmanager

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")  # 127.0.0.1: "localhost" tries ::1 first on Windows
MODEL = os.getenv("RERANK_MODEL", "gemma4:12b")
IS_WINDOWS = os.name == "nt"


def generate_json(prompt: str, schema: dict, model: str = MODEL) -> dict:
    """The answer is constrained to `schema` by Ollama itself (structured outputs), so it always parses."""
    body = {"model": model, "prompt": prompt, "format": schema, "stream": False, "think": False, "keep_alive": "3s",
            "options": {"temperature": 0, "num_ctx": 8192}}
    req = urllib.request.Request(OLLAMA_URL + "/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(json.load(r)["response"])


def _up() -> bool:
    try:
        urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2)
        return True
    except OSError:
        return False


@contextmanager
def ollama():
    """Start `ollama serve` only if it is down, and stop it only if we started it."""
    proc = None
    if not _up():
        proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(30):
            if _up():
                break
            time.sleep(1)
        else:
            raise RuntimeError("Ollama did not start")
    try:
        yield
    finally:
        if proc:
            stop(proc)


def stop(proc) -> None:
    """terminate() alone leaves llama-server holding ~8 GB of VRAM on Windows: kill the whole tree."""
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        proc.terminate()
