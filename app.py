import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse

# ── Config ──────────────────────────────────────────────────────────────────
LABELS_DIR = "userdata/labels"
OUTPUT_DIR = "userdata/dataset"
AUDIO_DIR = f"{OUTPUT_DIR}/audio_files"
CSV_PATH = f"{OUTPUT_DIR}/recorded_samples.csv"
SAMPLE_RATE = 16000

os.makedirs(AUDIO_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset — helpers
# ═══════════════════════════════════════════════════════════════════════════

def ensure_csv_header():
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("audio,text\n")


def append_to_csv(audio_path, text):
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        f.write(f'{audio_path},"{text}"\n')


def remove_csv_entry(audio_path):
    if not os.path.exists(CSV_PATH):
        return
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(CSV_PATH, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.startswith(audio_path + ","):
                f.write(line)


def load_recordings():
    recordings = {}
    if os.path.exists(CSV_PATH) and os.path.getsize(CSV_PATH) > 0:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    audio_path = parts[0].strip()
                    text = parts[1].strip().strip('"')
                    fn = os.path.basename(audio_path)
                    if text not in recordings:
                        recordings[text] = []
                    recordings[text].append({"filename": fn, "path": audio_path})
    return recordings


def validate_label_filename(filename):
    if not filename or ".." in filename or "/" in filename or "\\" in filename:
        return False
    return bool(re.match(r'^[\w\-. ]+\.txt$', filename))


ensure_csv_header()


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


# ── Label management endpoints ────────────────────────────────────────────

@app.get("/api/labels")
def list_labels():
    os.makedirs(LABELS_DIR, exist_ok=True)
    files = []
    for txt_file in sorted(os.listdir(LABELS_DIR)):
        if txt_file.endswith(".txt"):
            filepath = os.path.join(LABELS_DIR, txt_file)
            with open(filepath, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
            files.append({"filename": txt_file, "sentences": sentences})
    return {"files": files}


@app.post("/api/labels")
def create_label_file(body: dict):
    filename = body.get("filename", "").strip()
    if not filename.endswith(".txt"):
        filename += ".txt"
    if not validate_label_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    filepath = os.path.join(LABELS_DIR, filename)
    if os.path.exists(filepath):
        return JSONResponse(status_code=409, content={"error": "File already exists."})
    os.makedirs(LABELS_DIR, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        pass
    return {"filename": filename}


@app.delete("/api/labels/{filename}")
def delete_label_file(filename: str):
    if not validate_label_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    filepath = os.path.join(LABELS_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "File not found."})
    os.remove(filepath)
    return {"message": f"Deleted {filename}"}


@app.post("/api/labels/{filename}/add")
def add_sentence(filename: str, body: dict):
    if not validate_label_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    sentence = body.get("sentence", "").strip()
    if not sentence:
        return JSONResponse(status_code=400, content={"error": "Sentence is empty."})
    filepath = os.path.join(LABELS_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "File not found."})
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(sentence + "\n")
    return {"message": "Sentence added."}


@app.post("/api/labels/{filename}/remove")
def remove_sentence(filename: str, body: dict):
    if not validate_label_filename(filename):
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    sentence = body.get("sentence", "").strip()
    filepath = os.path.join(LABELS_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "File not found."})
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    found = False
    new_lines = []
    for line in lines:
        if line == sentence and not found:
            found = True
            continue
        new_lines.append(line)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")
    return {"message": "Sentence removed."}


# ── Dataset endpoints ─────────────────────────────────────────────────────

@app.get("/api/dataset")
def get_dataset():
    recordings = load_recordings()
    os.makedirs(LABELS_DIR, exist_ok=True)
    groups = []
    for txt_file in sorted(os.listdir(LABELS_DIR)):
        if txt_file.endswith(".txt"):
            filepath = os.path.join(LABELS_DIR, txt_file)
            with open(filepath, "r", encoding="utf-8") as f:
                sentences = [line.strip() for line in f if line.strip()]
            sentence_data = []
            for s in sentences:
                sentence_data.append({
                    "text": s,
                    "recordings": recordings.get(s, []),
                })
            groups.append({"filename": txt_file, "sentences": sentence_data})
    return {"groups": groups}


@app.get("/api/audio/{filename}")
def serve_audio(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    filepath = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Audio file not found."})
    return FileResponse(filepath, media_type="audio/wav")


@app.post("/api/recordings")
async def save_recording(
    audio: UploadFile = File(...),
    sentence: str = Form(...),
    replace: str = Form(None),
):
    audio_filename = f"sample_{int(time.time() * 1000)}.wav"
    audio_path = f"{AUDIO_DIR}/{audio_filename}"

    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    if replace:
        old_path = f"{AUDIO_DIR}/{replace}"
        remove_csv_entry(old_path)
        if os.path.exists(old_path):
            os.remove(old_path)

    append_to_csv(audio_path, sentence)
    return {"filename": audio_filename, "path": audio_path}


@app.delete("/api/recordings/{filename}")
def delete_recording(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    audio_path = f"{AUDIO_DIR}/{filename}"
    remove_csv_entry(audio_path)
    if os.path.exists(audio_path):
        os.remove(audio_path)
        return {"message": f"Deleted {filename}"}
    return JSONResponse(status_code=404, content={"error": "File not found."})


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
