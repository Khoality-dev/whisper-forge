import csv
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import zipfile
from glob import glob
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from starlette.background import BackgroundTask
from starlette.responses import FileResponse

# ── Config ──────────────────────────────────────────────────────────────────
DATASETS_DIR = "userdata/datasets"
MODELS_DIR = "userdata/models"
SAMPLE_RATE = 16000

os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_name(name: str) -> bool:
    """Check that a dataset/model name is safe for filesystem use."""
    return bool(name) and ".." not in name and "/" not in name and "\\" not in name


# ── Per-dataset CSV helpers ───────────────────────────────────────────────

def _labels_csv(ds_path: str) -> str:
    return os.path.join(ds_path, "labels.csv")


def _recordings_csv(ds_path: str) -> str:
    return os.path.join(ds_path, "recorded_samples.csv")


def _audio_dir(ds_path: str) -> str:
    return os.path.join(ds_path, "audio_files")


def _read_labels(ds_path: str) -> list[dict]:
    path = _labels_csv(ds_path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        return [{"id": int(r["id"]), "language": r["language"], "text": r["text"]} for r in reader]


def _write_labels(ds_path: str, labels: list[dict]):
    with open(_labels_csv(ds_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "language", "text"],
                                delimiter="|", quoting=csv.QUOTE_NONE)
        writer.writeheader()
        for label in labels:
            writer.writerow(label)


def _read_recordings(ds_path: str) -> list[dict]:
    path = _recordings_csv(ds_path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        return [{"audio": r["audio"], "label_id": int(r["label_id"])} for r in reader]


def _write_recordings(ds_path: str, recordings: list[dict]):
    with open(_recordings_csv(ds_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "label_id"],
                                delimiter="|", quoting=csv.QUOTE_NONE)
        writer.writeheader()
        for rec in recordings:
            writer.writerow(rec)


def _next_label_id(labels: list[dict]) -> int:
    if not labels:
        return 1
    return max(l["id"] for l in labels) + 1


def _init_dataset(ds_path: str):
    """Create dataset directory with empty CSVs and audio_files/."""
    os.makedirs(_audio_dir(ds_path), exist_ok=True)
    if not os.path.exists(_labels_csv(ds_path)):
        _write_labels(ds_path, [])
    if not os.path.exists(_recordings_csv(ds_path)):
        _write_recordings(ds_path, [])


def _add_sample(ds_path: str, wav_filename: str, text: str, language: str = ""):
    """Add a wav+text pair into the dataset CSVs. wav must already be in audio_files/."""
    labels = _read_labels(ds_path)
    recordings = _read_recordings(ds_path)
    new_id = _next_label_id(labels)
    labels.append({"id": new_id, "language": language, "text": text})
    audio_path = os.path.join(_audio_dir(ds_path), wav_filename)
    recordings.append({"audio": audio_path, "label_id": new_id})
    _write_labels(ds_path, labels)
    _write_recordings(ds_path, recordings)
    return new_id


# ── Dataset / model listing ──────────────────────────────────────────────

def _scan_samples(ds_path: str) -> list[dict]:
    """Join labels + recordings CSVs, return flat sample list."""
    labels = _read_labels(ds_path)
    recordings = _read_recordings(ds_path)
    id_to_text = {l["id"]: l["text"] for l in labels}
    samples = []
    for rec in recordings:
        text = id_to_text.get(rec["label_id"], "")
        filename = os.path.basename(rec["audio"])
        samples.append({"filename": filename, "text": text, "label_id": rec["label_id"]})
    return samples


def _list_datasets() -> list[dict]:
    datasets = []
    if not os.path.isdir(DATASETS_DIR):
        return datasets
    for entry in sorted(os.listdir(DATASETS_DIR)):
        ds_path = os.path.join(DATASETS_DIR, entry)
        if os.path.isdir(ds_path):
            count = len(_read_recordings(ds_path))
            datasets.append({"name": entry, "count": count})
    return datasets


def _list_models() -> list[dict]:
    models = []
    if not os.path.isdir(MODELS_DIR):
        return models
    for entry in sorted(os.listdir(MODELS_DIR)):
        model_path = os.path.join(MODELS_DIR, entry)
        if os.path.isdir(model_path):
            has_model = os.path.exists(os.path.join(model_path, "model.safetensors")) or \
                        os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
            status = "untrained"
            with train_lock:
                proc = train_processes.get(entry)
                if proc is not None and proc.poll() is None:
                    status = "training"
                elif has_model:
                    status = "trained"
            models.append({"name": entry, "status": status})
    return models


# ═══════════════════════════════════════════════════════════════════════════
#  Migration — convert old single-dataset format to multi-dataset
# ═══════════════════════════════════════════════════════════════════════════

def _migrate_v2():
    """If old userdata/dataset/ exists, move it to userdata/datasets/default/."""
    old_dir = "userdata/dataset"
    if not os.path.isdir(old_dir):
        return
    target = os.path.join(DATASETS_DIR, "default")
    if os.path.exists(target):
        return
    shutil.move(old_dir, target)
    print(f"[migrate] Moved {old_dir} -> {target}")


_migrate_v2()


# ═══════════════════════════════════════════════════════════════════════════
#  Train — helpers & state (per-model)
# ═══════════════════════════════════════════════════════════════════════════

train_processes: dict[str, subprocess.Popen] = {}
train_lock = threading.Lock()
train_log_lines: dict[str, list[str]] = {}


def _read_train_output(model_name: str):
    with train_lock:
        proc = train_processes.get(model_name)
    if proc is None:
        return
    for line in iter(proc.stdout.readline, ""):
        if not line:
            break
        with train_lock:
            if model_name not in train_log_lines:
                train_log_lines[model_name] = []
            train_log_lines[model_name].append(line)
    proc.wait()


# ═══════════════════════════════════════════════════════════════════════════
#  FastAPI App
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(title="WhisperForge")


# ── Dataset endpoints ─────────────────────────────────────────────────────

@app.get("/api/datasets")
def list_datasets():
    return {"datasets": _list_datasets()}


@app.post("/api/datasets")
def create_dataset(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "Dataset name is empty."})
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid dataset name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    if os.path.exists(ds_path):
        return JSONResponse(status_code=409, content={"error": "Dataset already exists."})
    _init_dataset(ds_path)
    return {"name": name, "count": 0}


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid dataset name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(ds_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found."})
    shutil.rmtree(ds_path)
    return {"message": f"Deleted dataset '{name}'."}


@app.get("/api/datasets/{name}/download")
def download_dataset(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid dataset name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(ds_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found."})
    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, name)
    shutil.make_archive(zip_path, "zip", ".", ds_path)
    return FileResponse(
        zip_path + ".zip",
        media_type="application/zip",
        filename=f"{name}.zip",
        background=BackgroundTask(shutil.rmtree, tmp),
    )


@app.get("/api/datasets/{name}")
def get_dataset_samples(
    name: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid dataset name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(ds_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found."})
    all_samples = _scan_samples(ds_path)
    total = len(all_samples)
    page = all_samples[offset:offset + limit]
    return {"samples": page, "total": total}


@app.post("/api/datasets/{name}/recordings")
async def save_recording(
    name: str,
    audio: UploadFile = File(...),
    text: str = Form(...),
    replace: str = Form(""),
):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid dataset name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(ds_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found."})

    wav_filename = f"sample_{int(time.time() * 1000)}.wav"
    wav_path = os.path.join(_audio_dir(ds_path), wav_filename)

    content = await audio.read()
    with open(wav_path, "wb") as f:
        f.write(content)

    _add_sample(ds_path, wav_filename, text.strip(), language=name)

    # Delete old sample if replacing
    if replace:
        _delete_sample_by_filename(ds_path, replace)

    return {"filename": wav_filename}


def _delete_sample_by_filename(ds_path: str, wav_filename: str):
    """Remove a sample from CSVs and delete audio file."""
    audio_path = os.path.join(_audio_dir(ds_path), wav_filename)
    labels = _read_labels(ds_path)
    recordings = _read_recordings(ds_path)

    # Find label_id for this audio
    label_ids_to_remove = set()
    new_recordings = []
    for rec in recordings:
        if os.path.basename(rec["audio"]) == wav_filename:
            label_ids_to_remove.add(rec["label_id"])
        else:
            new_recordings.append(rec)

    # Only remove labels that have no other recordings
    remaining_label_ids = {r["label_id"] for r in new_recordings}
    labels = [l for l in labels if l["id"] not in label_ids_to_remove or l["id"] in remaining_label_ids]

    _write_labels(ds_path, labels)
    _write_recordings(ds_path, new_recordings)

    if os.path.exists(audio_path):
        os.remove(audio_path)


@app.delete("/api/datasets/{name}/recordings/{filename}")
def delete_recording(name: str, filename: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    audio_path = os.path.join(_audio_dir(ds_path), filename)
    if not os.path.exists(audio_path):
        return JSONResponse(status_code=404, content={"error": "Sample not found."})
    _delete_sample_by_filename(ds_path, filename)
    return {"message": f"Deleted sample '{filename}'."}


@app.post("/api/datasets/{name}/upload")
async def upload_to_dataset(name: str, files: list[UploadFile] = File(...)):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid dataset name."})
    ds_path = os.path.join(DATASETS_DIR, name)
    if not os.path.isdir(ds_path):
        return JSONResponse(status_code=404, content={"error": "Dataset not found."})

    added = 0

    def _import_pair(wav_src_path: str, txt_src_path: str | None, stem: str):
        """Copy wav to audio_files/ and add CSV entry."""
        nonlocal added
        dest_wav = os.path.join(_audio_dir(ds_path), f"{stem}.wav")
        shutil.copy2(wav_src_path, dest_wav)
        text = ""
        if txt_src_path and os.path.exists(txt_src_path):
            with open(txt_src_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        _add_sample(ds_path, f"{stem}.wav", text, language=name)
        added += 1

    # Collect loose txt files for pairing with loose wav files
    loose_txts = {}

    for upload in files:
        content = await upload.read()
        fname = upload.filename or ""

        if fname.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                with tempfile.TemporaryDirectory() as tmp:
                    zf.extractall(tmp)
                    for root, dirs, fnames in os.walk(tmp):
                        for f in fnames:
                            if f.lower().endswith(".wav"):
                                stem = os.path.splitext(f)[0]
                                wav_src = os.path.join(root, f)
                                txt_src = os.path.join(root, f"{stem}.txt")
                                _import_pair(wav_src, txt_src if os.path.exists(txt_src) else None, stem)
        elif fname.lower().endswith(".txt"):
            # Stash txt content for pairing
            stem = os.path.splitext(fname)[0]
            loose_txts[stem] = content
        elif fname.lower().endswith(".wav"):
            stem = os.path.splitext(fname)[0]
            # Save wav to audio_files/
            dest_wav = os.path.join(_audio_dir(ds_path), fname)
            with open(dest_wav, "wb") as f:
                f.write(content)
            # Check if we already have its txt
            text = ""
            if stem in loose_txts:
                text = loose_txts.pop(stem).decode("utf-8", errors="replace").strip()
            _add_sample(ds_path, fname, text, language=name)
            added += 1

    # Process remaining loose txt files that arrived before their wav
    # (pair with wav files that were already written but had no txt)
    # This handles the case where txt arrives after wav in the file list
    for stem, txt_content in loose_txts.items():
        wav_path = os.path.join(_audio_dir(ds_path), f"{stem}.wav")
        if os.path.exists(wav_path):
            text = txt_content.decode("utf-8", errors="replace").strip()
            # Update the label text for this audio
            labels = _read_labels(ds_path)
            recordings = _read_recordings(ds_path)
            for rec in recordings:
                if os.path.basename(rec["audio"]) == f"{stem}.wav":
                    for label in labels:
                        if label["id"] == rec["label_id"]:
                            label["text"] = text
                            break
            _write_labels(ds_path, labels)

    return {"added": added}


@app.get("/api/audio/{dataset}/{filename}")
def serve_audio(dataset: str, filename: str):
    if not _safe_name(dataset) or ".." in filename:
        return JSONResponse(status_code=400, content={"error": "Invalid path."})
    filepath = os.path.join(DATASETS_DIR, dataset, "audio_files", filename)
    if not os.path.exists(filepath):
        return JSONResponse(status_code=404, content={"error": "Audio file not found."})
    return FileResponse(filepath, media_type="audio/wav")


# ── Model version endpoints ──────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    return {"models": _list_models()}


@app.post("/api/models")
def create_model(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse(status_code=400, content={"error": "Model name is empty."})
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    model_path = os.path.join(MODELS_DIR, name)
    if os.path.exists(model_path):
        return JSONResponse(status_code=409, content={"error": "Model version already exists."})
    os.makedirs(model_path)
    config = {
        "datasets": [],
        "epochs": 5,
        "learning_rate": 1e-5,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "fp16": False,
        "logging_steps": 100,
        "save_steps": 500,
        "eval_steps": 500,
    }
    with open(os.path.join(model_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return {"name": name, "status": "untrained"}


@app.delete("/api/models/{name}")
def delete_model(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    model_path = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(model_path):
        return JSONResponse(status_code=404, content={"error": "Model version not found."})
    with train_lock:
        proc = train_processes.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
        train_processes.pop(name, None)
        train_log_lines.pop(name, None)
    shutil.rmtree(model_path)
    return {"message": f"Deleted model version '{name}'."}


@app.get("/api/models/{name}/config")
def get_model_config(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    config_path = os.path.join(MODELS_DIR, name, "config.json")
    if not os.path.exists(config_path):
        return JSONResponse(status_code=404, content={"error": "Model version not found."})
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.put("/api/models/{name}/config")
def save_model_config(name: str, config: dict):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    model_path = os.path.join(MODELS_DIR, name)
    if not os.path.isdir(model_path):
        return JSONResponse(status_code=404, content={"error": "Model version not found."})
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return {"message": "Config saved."}


@app.post("/api/models/{name}/train")
def start_model_training(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    model_path = os.path.join(MODELS_DIR, name)
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return JSONResponse(status_code=404, content={"error": "Model version not found."})

    with train_lock:
        proc = train_processes.get(name)
        if proc is not None and proc.poll() is None:
            return JSONResponse(
                status_code=409,
                content={"error": "Training is already running for this model."},
            )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset_names = config.get("datasets", [])
    if not dataset_names:
        return JSONResponse(
            status_code=400,
            content={"error": "No datasets selected. Configure datasets first."},
        )

    dataset_dirs = []
    for ds_name in dataset_names:
        ds_path = os.path.join(DATASETS_DIR, ds_name)
        if not os.path.isdir(ds_path):
            return JSONResponse(
                status_code=400,
                content={"error": f"Dataset '{ds_name}' not found."},
            )
        dataset_dirs.append(ds_path)

    cmd = [
        sys.executable, "train.py",
        "--dataset_dirs", *dataset_dirs,
        "--output_dir", model_path,
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
        train_log_lines[name] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        train_processes[name] = proc

    reader = threading.Thread(target=_read_train_output, args=(name,), daemon=True)
    reader.start()

    return {"message": f"Training started (PID {proc.pid})."}


@app.post("/api/models/{name}/stop")
def stop_model_training(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    with train_lock:
        proc = train_processes.get(name)
        if proc and proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=10)
            train_processes[name] = None
            return {"message": "Training stopped."}
    return {"message": "No training process running."}


@app.get("/api/models/{name}/status")
def model_train_status(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    with train_lock:
        proc = train_processes.get(name)
        log = "".join(train_log_lines.get(name, []))
    running = proc is not None and proc.poll() is None
    exit_code = None
    if proc is not None and proc.poll() is not None:
        exit_code = proc.returncode
    return {"running": running, "log": log, "exit_code": exit_code}


@app.get("/api/models/{name}/download")
def download_model(name: str):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    model_path = os.path.join(MODELS_DIR, name)
    has_model = os.path.exists(os.path.join(model_path, "model.safetensors")) or \
                os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
    if not has_model:
        return JSONResponse(
            status_code=404,
            content={"error": "No trained model found. Train a model first."},
        )
    tmp = tempfile.mkdtemp()
    zip_path = os.path.join(tmp, f"whisper-{name}")
    shutil.make_archive(zip_path, "zip", ".", model_path)
    return FileResponse(
        zip_path + ".zip",
        media_type="application/zip",
        filename=f"whisper-{name}.zip",
        background=BackgroundTask(shutil.rmtree, tmp),
    )


@app.post("/api/models/{name}/predict")
def predict_model_samples(name: str, body: dict = {}):
    if not _safe_name(name):
        return JSONResponse(status_code=400, content={"error": "Invalid model name."})
    model_path = os.path.join(MODELS_DIR, name)
    has_model = os.path.exists(os.path.join(model_path, "model.safetensors")) or \
                os.path.exists(os.path.join(model_path, "pytorch_model.bin"))
    if not has_model:
        return JSONResponse(
            status_code=404,
            content={"error": "No trained model found. Train a model first."},
        )

    config_path = os.path.join(model_path, "config.json")
    dataset_dirs = []
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        for ds_name in config.get("datasets", []):
            ds_path = os.path.join(DATASETS_DIR, ds_name)
            if os.path.isdir(ds_path):
                dataset_dirs.append(ds_path)

    if not dataset_dirs:
        return JSONResponse(
            status_code=400,
            content={"error": "No valid datasets found for this model."},
        )

    n = int(body.get("n", 5))
    result = subprocess.run(
        [
            sys.executable, "predict.py",
            "--model_dir", model_path,
            "--dataset_dirs", *dataset_dirs,
            "--n", str(n),
        ],
        capture_output=True, text=True, timeout=120,
    )

    if result.returncode != 0:
        return JSONResponse(
            status_code=500,
            content={"error": result.stderr or "Prediction failed."},
        )

    return json.loads(result.stdout)


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
