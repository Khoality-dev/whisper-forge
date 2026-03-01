import csv
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = "userdata/dataset"
AUDIO_DIR = f"{OUTPUT_DIR}/audio_files"
LABELS_CSV = f"{OUTPUT_DIR}/labels.csv"
RECORDINGS_CSV = f"{OUTPUT_DIR}/recorded_samples.csv"
SAMPLE_RATE = 16000

# Legacy paths (for migration)
LEGACY_LABELS_DIR = "userdata/labels"
LEGACY_CSV_PATH = f"{OUTPUT_DIR}/recorded_samples.csv"

os.makedirs(AUDIO_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV helpers — labels.csv
# ═══════════════════════════════════════════════════════════════════════════

def _read_labels():
    """Return list of dicts: [{id, language, text}, ...]"""
    if not os.path.exists(LABELS_CSV) or os.path.getsize(LABELS_CSV) == 0:
        return []
    with open(LABELS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        rows = []
        for row in reader:
            rows.append({
                "id": int(row["id"]),
                "language": row["language"],
                "text": row["text"],
            })
        return rows


def _write_labels(labels):
    """Write list of label dicts to labels.csv."""
    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    with open(LABELS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "language", "text"],
            delimiter="|", quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        for label in labels:
            writer.writerow(label)


def _next_label_id(labels):
    """Return next available label ID."""
    if not labels:
        return 1
    return max(l["id"] for l in labels) + 1


# ═══════════════════════════════════════════════════════════════════════════
#  CSV helpers — recorded_samples.csv
# ═══════════════════════════════════════════════════════════════════════════

def _read_recordings():
    """Return list of dicts: [{audio, label_id}, ...]"""
    if not os.path.exists(RECORDINGS_CSV) or os.path.getsize(RECORDINGS_CSV) == 0:
        return []
    with open(RECORDINGS_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        rows = []
        for row in reader:
            rows.append({
                "audio": row["audio"],
                "label_id": int(row["label_id"]),
            })
        return rows


def _write_recordings(recordings):
    """Write list of recording dicts to recorded_samples.csv."""
    os.makedirs(os.path.dirname(RECORDINGS_CSV), exist_ok=True)
    with open(RECORDINGS_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["audio", "label_id"],
            delimiter="|", quoting=csv.QUOTE_NONE,
        )
        writer.writeheader()
        for rec in recordings:
            writer.writerow(rec)


# ═══════════════════════════════════════════════════════════════════════════
#  Migration — convert legacy label files + old CSV to new format
# ═══════════════════════════════════════════════════════════════════════════

def _migrate_legacy():
    """Convert old label .txt files + old recorded_samples.csv into new format.

    Skips if labels.csv already exists.
    """
    if os.path.exists(LABELS_CSV):
        return

    labels = []
    text_to_id = {}
    next_id = 1

    # Read old label files
    if os.path.isdir(LEGACY_LABELS_DIR):
        for txt_file in sorted(os.listdir(LEGACY_LABELS_DIR)):
            if not txt_file.endswith(".txt"):
                continue
            language = txt_file[:-4]  # strip .txt
            filepath = os.path.join(LEGACY_LABELS_DIR, txt_file)
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    labels.append({
                        "id": next_id,
                        "language": language,
                        "text": text,
                    })
                    # Map (language, text) -> id for recording matching
                    text_to_id[text] = next_id
                    next_id += 1

    # Build a normalized-text lookup for fuzzy matching (smart quotes, etc.)
    def _normalize(s):
        return (s
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .strip().rstrip('"'))

    norm_to_id = {}
    for text, lid in text_to_id.items():
        norm_to_id[_normalize(text)] = lid

    # Read old recordings CSV and remap text -> label_id
    recordings = []
    if os.path.exists(LEGACY_CSV_PATH) and os.path.getsize(LEGACY_CSV_PATH) > 0:
        with open(LEGACY_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header and len(header) >= 2 and header[0].strip() == "audio" and header[1].strip() == "text":
                for row in reader:
                    if len(row) < 2:
                        continue
                    audio_path = row[0].strip()
                    text = row[1].strip()
                    label_id = text_to_id.get(text)
                    if label_id is None:
                        label_id = norm_to_id.get(_normalize(text))
                    if label_id is not None:
                        recordings.append({
                            "audio": audio_path,
                            "label_id": label_id,
                        })

    if not labels and not recordings:
        return

    _write_labels(labels)
    _write_recordings(recordings)
    print(f"[migrate] Converted {len(labels)} labels and {len(recordings)} recordings to new format.")



# ── Run migration on startup ──
_migrate_legacy()


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


# ── Dataset endpoint ─────────────────────────────────────────────────────

@app.get("/api/dataset")
def get_dataset():
    labels = _read_labels()
    recordings = _read_recordings()

    # Group recordings by label_id
    recs_by_label = {}
    for rec in recordings:
        recs_by_label.setdefault(rec["label_id"], []).append({
            "filename": os.path.basename(rec["audio"]),
            "path": rec["audio"],
        })

    # Group labels by language
    lang_groups = {}
    for label in labels:
        lang = label["language"]
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append({
            "id": label["id"],
            "text": label["text"],
            "recordings": recs_by_label.get(label["id"], []),
        })

    groups = []
    for lang in sorted(lang_groups.keys()):
        groups.append({
            "language": lang,
            "sentences": lang_groups[lang],
        })

    return {"groups": groups}


# ── Language management ──────────────────────────────────────────────────

@app.post("/api/languages")
def create_language(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "Language name is empty."})
    # Language is created implicitly with the first label — this is a no-op
    return {"language": name}


@app.delete("/api/languages/{name}")
def delete_language(name: str):
    labels = _read_labels()
    recordings = _read_recordings()

    # Find label IDs for this language
    ids_to_remove = {l["id"] for l in labels if l["language"] == name}

    if not ids_to_remove:
        return JSONResponse(status_code=404, content={"error": "Language not found."})

    # Delete audio files for removed recordings
    for rec in recordings:
        if rec["label_id"] in ids_to_remove:
            audio_path = rec["audio"]
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Filter out labels and recordings for this language
    labels = [l for l in labels if l["language"] != name]
    recordings = [r for r in recordings if r["label_id"] not in ids_to_remove]

    _write_labels(labels)
    _write_recordings(recordings)

    return {"message": f"Deleted language '{name}' and all associated data."}


# ── Label management ─────────────────────────────────────────────────────

@app.post("/api/labels/add")
def add_label(body: dict):
    language = body.get("language", "").strip()
    text = body.get("text", "").strip()
    if not language:
        return JSONResponse(status_code=400, content={"error": "Language is empty."})
    if not text:
        return JSONResponse(status_code=400, content={"error": "Text is empty."})

    labels = _read_labels()
    new_id = _next_label_id(labels)
    labels.append({"id": new_id, "language": language, "text": text})
    _write_labels(labels)

    return {"id": new_id, "message": "Label added."}


@app.post("/api/labels/remove")
def remove_label(body: dict):
    label_id = body.get("id")
    if label_id is None:
        return JSONResponse(status_code=400, content={"error": "Label ID is required."})
    label_id = int(label_id)

    labels = _read_labels()
    recordings = _read_recordings()

    # Check label exists
    if not any(l["id"] == label_id for l in labels):
        return JSONResponse(status_code=404, content={"error": "Label not found."})

    # Delete audio files for this label's recordings
    for rec in recordings:
        if rec["label_id"] == label_id:
            audio_path = rec["audio"]
            if os.path.exists(audio_path):
                os.remove(audio_path)

    # Remove label and its recordings
    labels = [l for l in labels if l["id"] != label_id]
    recordings = [r for r in recordings if r["label_id"] != label_id]

    _write_labels(labels)
    _write_recordings(recordings)

    return {"message": "Label and associated recordings removed."}


# ── Recording endpoints ──────────────────────────────────────────────────

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
    label_id: int = Form(...),
    replace: str = Form(""),
):
    audio_filename = f"sample_{int(time.time() * 1000)}.wav"
    audio_path = f"{AUDIO_DIR}/{audio_filename}"

    content = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(content)

    recordings = _read_recordings()

    if replace:
        old_path = f"{AUDIO_DIR}/{replace}"
        recordings = [r for r in recordings if r["audio"] != old_path]
        if os.path.exists(old_path):
            os.remove(old_path)

    recordings.append({"audio": audio_path, "label_id": label_id})
    _write_recordings(recordings)

    return {"filename": audio_filename, "path": audio_path}


@app.delete("/api/recordings/{filename}")
def delete_recording(filename: str):
    if ".." in filename or "/" in filename or "\\" in filename:
        return JSONResponse(status_code=400, content={"error": "Invalid filename."})
    audio_path = f"{AUDIO_DIR}/{filename}"

    recordings = _read_recordings()
    new_recordings = [r for r in recordings if r["audio"] != audio_path]

    if len(new_recordings) == len(recordings):
        return JSONResponse(status_code=404, content={"error": "Recording not found."})

    _write_recordings(new_recordings)
    if os.path.exists(audio_path):
        os.remove(audio_path)

    return {"message": f"Deleted {filename}"}


@app.get("/api/dataset/count")
def dataset_count():
    return {"count": len(_read_recordings())}


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
        "--labels", LABELS_CSV,
        "--recordings", RECORDINGS_CSV,
        "--output_dir", config.get("output_dir", "userdata/outputs"),
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


@app.get("/api/train/download")
def download_model():
    model_dir = "userdata/outputs"
    if not os.path.isdir(model_dir):
        return JSONResponse(
            status_code=404,
            content={"error": "No trained model found. Train a model first."},
        )

    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, "whisper-finetuned")
    shutil.make_archive(zip_path, "zip", ".", model_dir)

    return FileResponse(
        zip_path + ".zip",
        media_type="application/zip",
        filename="whisper-finetuned.zip",
        background=BackgroundTask(shutil.rmtree, tmp),
    )


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
