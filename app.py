import os
import random
import subprocess
import sys
import time
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

# ── Config ──────────────────────────────────────────────────────────────────
LABELS_DIR = "userdata/labels"
OUTPUT_DIR = "userdata/dataset"
AUDIO_DIR = f"{OUTPUT_DIR}/audio_files"
CSV_PATH = f"{OUTPUT_DIR}/recorded_samples.csv"
SAMPLE_RATE = 16000

os.makedirs(AUDIO_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Collect Data — helpers & state
# ═══════════════════════════════════════════════════════════════════════════

def load_sentences():
    os.makedirs(LABELS_DIR, exist_ok=True)
    sentences = []
    for txt_file in sorted(os.listdir(LABELS_DIR)):
        if txt_file.endswith(".txt"):
            with open(os.path.join(LABELS_DIR, txt_file), "r", encoding="utf-8") as f:
                sentences.extend(line.strip() for line in f if line.strip())
    return sentences


def load_completed_sentences():
    completed = set()
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    completed.add(parts[1].strip().strip('"'))
    return completed


def ensure_csv_header():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("audio,text\n")


def append_to_csv(audio_path, text):
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        f.write(f'{audio_path},"{text}"\n')


class CollectState:
    def __init__(self):
        self.all_sentences = load_sentences()
        self.completed = load_completed_sentences()
        self.pending = [s for s in self.all_sentences if s not in self.completed]
        random.shuffle(self.pending)
        self.current_index = 0

    @property
    def total(self):
        return len(self.all_sentences)

    @property
    def done_count(self):
        return len(self.completed)

    @property
    def current_sentence(self):
        if self.current_index < len(self.pending):
            return self.pending[self.current_index]
        return None

    def advance(self):
        self.current_index += 1

    def mark_done(self, sentence):
        self.completed.add(sentence)


ensure_csv_header()
collect = CollectState()


# ═══════════════════════════════════════════════════════════════════════════
#  Train — helpers & state
# ═══════════════════════════════════════════════════════════════════════════

train_process: subprocess.Popen | None = None
train_lock = threading.Lock()
train_log_lines: list[str] = []


def _read_train_output():
    """Background thread that reads subprocess stdout into train_log_lines."""
    global train_process
    with train_lock:
        proc = train_process
    if proc is None:
        return
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        with train_lock:
            train_log_lines.append(line)
    proc.wait()


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="WhisperForge")


# ── Collect Data endpoints ────────────────────────────────────────────────

def _sentence_response():
    sentence = collect.current_sentence
    return {
        "sentence": sentence if sentence else "All sentences have been recorded!",
        "progress": {"done": collect.done_count, "total": collect.total},
        "finished": sentence is None,
    }


@app.get("/api/sentences/current")
def get_current_sentence():
    return _sentence_response()


@app.post("/api/sentences/skip")
def skip_sentence():
    collect.advance()
    return _sentence_response()


@app.post("/api/recordings")
async def save_recording(audio: UploadFile = File(...), sentence: str = Form(...)):
    audio_path = f"{AUDIO_DIR}/sample_{int(time.time() * 1000)}.wav"

    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    append_to_csv(audio_path, sentence)
    collect.mark_done(sentence)
    collect.advance()

    return _sentence_response()


# ── Train endpoints ──────────────────────────────────────────────────────

@app.post("/api/train/start")
def start_training(config: dict):
    global train_process, train_log_lines

    with train_lock:
        if train_process is not None and train_process.poll() is None:
            return JSONResponse(
                status_code=409,
                content={"error": "Training is already running."},
            )

    cmd = [
        sys.executable, "train.py",
        "--csv", CSV_PATH,
        "--output_dir", config.get("output_dir", "whisper-finetuned"),
        "--lang", config.get("language", "en"),
        "--epochs", str(int(config.get("epochs", 5))),
        "--learning_rate", str(config.get("learning_rate", 1e-5)),
        "--train_batch_size", str(int(config.get("train_batch_size", 8))),
        "--eval_batch_size", str(int(config.get("eval_batch_size", 8))),
        "--logging_steps", str(int(config.get("logging_steps", 100))),
        "--save_steps", str(int(config.get("save_steps", 500))),
        "--eval_steps", str(int(config.get("eval_steps", 500))),
    ]
    if config.get("fp16", False):
        cmd.append("--fp16")

    with train_lock:
        train_log_lines = []
        train_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    reader = threading.Thread(target=_read_train_output, daemon=True)
    reader.start()

    return {"message": f"Training started (PID {train_process.pid})."}


@app.post("/api/train/stop")
def stop_training():
    global train_process
    with train_lock:
        if train_process and train_process.poll() is None:
            train_process.terminate()
            train_process.wait(timeout=10)
            train_process = None
            return {"message": "Training stopped."}
    return {"message": "No training process running."}


@app.get("/api/train/status")
def train_status():
    with train_lock:
        proc = train_process
        log = "".join(train_log_lines)

    running = proc is not None and proc.poll() is None
    exit_code = None
    if proc is not None and proc.poll() is not None:
        exit_code = proc.returncode

    return {"running": running, "log": log, "exit_code": exit_code}


# ── Static file serving (React SPA) ──────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
