import os
import random
import subprocess
import sys
import time
import threading

import gradio as gr
import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Config ──────────────────────────────────────────────────────────────────
LABELS_DIR = "userdata/labels"
OUTPUT_DIR = "userdata/dataset"
AUDIO_DIR = f"{OUTPUT_DIR}/audio_files"
CSV_PATH = f"{OUTPUT_DIR}/recorded_samples.csv"
SAMPLE_RATE = 16000
BLOCK_SIZE = 480

os.makedirs(AUDIO_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Collect Data — helpers & state
# ═══════════════════════════════════════════════════════════════════════════

def load_sentences():
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
        with open(CSV_PATH, "w", encoding="utf-8") as f:
            f.write("audio,text\n")


def append_to_csv(audio_path, text):
    with open(CSV_PATH, "a", encoding="utf-8") as f:
        f.write(f'{audio_path},"{text}"\n')


class RecorderState:
    def __init__(self):
        self.is_recording = False
        self.audio_data: list[np.ndarray] = []
        self.thread: threading.Thread | None = None
        self.last_audio: np.ndarray | None = None

    def start(self):
        self.is_recording = True
        self.audio_data = []
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_recording = False
        if self.thread:
            self.thread.join()
            self.thread = None
        if self.audio_data:
            self.last_audio = np.concatenate(self.audio_data)
        else:
            self.last_audio = None

    def _capture(self):
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK_SIZE
        ) as stream:
            while self.is_recording:
                block, overflowed = stream.read(BLOCK_SIZE)
                self.audio_data.append(block.copy())

    def save_wav(self, path):
        if self.last_audio is not None:
            sf.write(path, self.last_audio, SAMPLE_RATE)

    def get_playback_tuple(self):
        if self.last_audio is not None:
            return (SAMPLE_RATE, self.last_audio.flatten())
        return None


class CollectState:
    def __init__(self):
        self.all_sentences = load_sentences()
        self.completed = load_completed_sentences()
        self.pending = [s for s in self.all_sentences if s not in self.completed]
        random.shuffle(self.pending)
        self.current_index = 0
        self.recorder = RecorderState()

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

    def progress_text(self):
        return f"{self.done_count} / {self.total} sentences recorded"

    def progress_ratio(self):
        return self.done_count / self.total if self.total > 0 else 0


ensure_csv_header()
collect = CollectState()


# ── Collect Data callbacks ──────────────────────────────────────────────────

def get_display():
    sentence = collect.current_sentence or "All sentences have been recorded!"
    return sentence, collect.progress_text(), collect.progress_ratio()


def on_record():
    collect.recorder.start()
    return (
        gr.update(interactive=False, visible=False),  # record btn
        gr.update(interactive=True, visible=True),     # stop btn
        gr.update(value=None),                         # audio player
        gr.update(interactive=False),                  # save btn
        gr.update(interactive=False),                  # re-record btn
        gr.update(interactive=False),                  # skip btn
        "Recording...",
    )


def on_stop():
    collect.recorder.stop()
    playback = collect.recorder.get_playback_tuple()
    return (
        gr.update(interactive=True, visible=True),     # record btn
        gr.update(interactive=False, visible=False),   # stop btn
        gr.update(value=playback),                     # audio player
        gr.update(interactive=True),                   # save btn
        gr.update(interactive=True),                   # re-record btn
        gr.update(interactive=True),                   # skip btn
        "Recording stopped. Listen back, then Save or Re-record.",
    )


def on_save():
    sentence = collect.current_sentence
    if sentence is None:
        return (*get_display(), None, "No sentence to save.")

    audio_path = f"{AUDIO_DIR}/sample_{int(time.time())}.wav"
    collect.recorder.save_wav(audio_path)
    append_to_csv(audio_path, sentence)
    collect.mark_done(sentence)
    collect.advance()

    sent, prog_text, prog_ratio = get_display()
    return (sent, prog_text, prog_ratio, None, "Saved! Moved to next sentence.")


def on_skip():
    collect.advance()
    sent, prog_text, prog_ratio = get_display()
    return (sent, prog_text, prog_ratio, None, "Skipped. Moved to next sentence.")


def on_rerecord():
    collect.recorder.last_audio = None
    return (
        gr.update(interactive=True, visible=True),     # record btn
        gr.update(interactive=False, visible=False),   # stop btn
        gr.update(value=None),                         # audio player
        gr.update(interactive=False),                  # save btn
        gr.update(interactive=False),                  # re-record btn
        gr.update(interactive=True),                   # skip btn
        "Discarded. Click Record to try again.",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Train — helpers & state
# ═══════════════════════════════════════════════════════════════════════════

train_process: subprocess.Popen | None = None
train_lock = threading.Lock()


def is_training():
    with train_lock:
        return train_process is not None and train_process.poll() is None


def on_start_training(language, epochs, lr, train_bs, eval_bs, fp16,
                      logging_steps, save_steps, eval_steps, output_dir):
    global train_process
    if is_training():
        return "Training is already running.", gr.update(), gr.update()

    cmd = [
        sys.executable, "train.py",
        "--csv", CSV_PATH,
        "--output_dir", output_dir,
        "--lang", language,
        "--epochs", str(int(epochs)),
        "--learning_rate", str(lr),
        "--train_batch_size", str(int(train_bs)),
        "--eval_batch_size", str(int(eval_bs)),
        "--logging_steps", str(int(logging_steps)),
        "--save_steps", str(int(save_steps)),
        "--eval_steps", str(int(eval_steps)),
    ]
    if fp16:
        cmd.append("--fp16")

    with train_lock:
        train_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    return (
        f"Training started (PID {train_process.pid}).\nCommand: {' '.join(cmd)}",
        gr.update(interactive=False),   # start btn
        gr.update(interactive=True),    # stop btn
    )


def on_stop_training():
    global train_process
    with train_lock:
        if train_process and train_process.poll() is None:
            train_process.terminate()
            train_process.wait(timeout=10)
            train_process = None
            return "Training stopped.", gr.update(interactive=True), gr.update(interactive=False)
    return "No training process running.", gr.update(), gr.update()


def on_refresh_log():
    global train_process
    with train_lock:
        proc = train_process
    if proc is None:
        return "No training process.", gr.update(), gr.update()

    # Read whatever is available without blocking
    lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            lines.append(line)
    except Exception:
        pass

    output = "".join(lines) if lines else "(no new output)"

    if proc.poll() is not None:
        # Process finished
        remaining = proc.stdout.read()
        if remaining:
            output += remaining
        rc = proc.returncode
        with train_lock:
            train_process = None
        return (
            output + f"\n\nTraining finished (exit code {rc}).",
            gr.update(interactive=True),    # start btn
            gr.update(interactive=False),   # stop btn
        )

    return output, gr.update(), gr.update()


# ═══════════════════════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════════════════════

with gr.Blocks(title="WhisperForge") as app:
    gr.Markdown("# WhisperForge")

    # ── Tab 1: Collect Data ─────────────────────────────────────────────
    with gr.Tab("Collect Data"):
        progress_label = gr.Textbox(
            value=collect.progress_text(), label="Progress", interactive=False,
        )
        progress_bar = gr.Slider(
            minimum=0, maximum=1, value=collect.progress_ratio(),
            interactive=False, show_label=False,
        )
        sentence_display = gr.Textbox(
            value=collect.current_sentence or "All sentences have been recorded!",
            label="Read this sentence aloud:",
            interactive=False, lines=3,
        )
        audio_player = gr.Audio(label="Playback", type="numpy", interactive=False)

        with gr.Row():
            record_btn = gr.Button("Record", variant="primary")
            stop_btn = gr.Button("Stop", variant="stop", interactive=False, visible=False)
            save_btn = gr.Button("Save", variant="primary", interactive=False)
            rerecord_btn = gr.Button("Re-record", interactive=False)
            skip_btn = gr.Button("Skip")

        collect_status = gr.Textbox(
            value="Ready. Click Record to start.", label="Status", interactive=False,
        )

        record_btn.click(
            fn=on_record,
            outputs=[record_btn, stop_btn, audio_player, save_btn, rerecord_btn, skip_btn, collect_status],
        )
        stop_btn.click(
            fn=on_stop,
            outputs=[record_btn, stop_btn, audio_player, save_btn, rerecord_btn, skip_btn, collect_status],
        )
        save_btn.click(
            fn=on_save,
            outputs=[sentence_display, progress_label, progress_bar, audio_player, collect_status],
        )
        skip_btn.click(
            fn=on_skip,
            outputs=[sentence_display, progress_label, progress_bar, audio_player, collect_status],
        )
        rerecord_btn.click(
            fn=on_rerecord,
            outputs=[record_btn, stop_btn, audio_player, save_btn, rerecord_btn, skip_btn, collect_status],
        )

    # ── Tab 2: Train ───────────────────────────────────────────────────
    with gr.Tab("Train"):
        gr.Markdown("Configure and launch Whisper fine-tuning. "
                     "Training runs `train.py` as a subprocess.")

        with gr.Row():
            with gr.Column():
                lang_input = gr.Textbox(value="en", label="Language code")
                epochs_input = gr.Number(value=5, label="Epochs", precision=0)
                lr_input = gr.Number(value=1e-5, label="Learning rate")
                fp16_input = gr.Checkbox(value=False, label="FP16 (mixed precision)")
            with gr.Column():
                train_bs_input = gr.Number(value=8, label="Train batch size", precision=0)
                eval_bs_input = gr.Number(value=8, label="Eval batch size", precision=0)
                logging_steps_input = gr.Number(value=100, label="Logging steps", precision=0)
                save_steps_input = gr.Number(value=500, label="Save steps", precision=0)
                eval_steps_input = gr.Number(value=500, label="Eval steps", precision=0)

        output_dir_input = gr.Textbox(value="whisper-finetuned", label="Output directory")

        with gr.Row():
            train_start_btn = gr.Button("Start Training", variant="primary")
            train_stop_btn = gr.Button("Stop Training", variant="stop", interactive=False)
            train_refresh_btn = gr.Button("Refresh Log")

        train_log = gr.Textbox(
            label="Training Log", interactive=False, lines=15, max_lines=30,
        )

        train_start_btn.click(
            fn=on_start_training,
            inputs=[lang_input, epochs_input, lr_input, train_bs_input, eval_bs_input,
                    fp16_input, logging_steps_input, save_steps_input, eval_steps_input,
                    output_dir_input],
            outputs=[train_log, train_start_btn, train_stop_btn],
        )
        train_stop_btn.click(
            fn=on_stop_training,
            outputs=[train_log, train_start_btn, train_stop_btn],
        )
        train_refresh_btn.click(
            fn=on_refresh_log,
            outputs=[train_log, train_start_btn, train_stop_btn],
        )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0")
