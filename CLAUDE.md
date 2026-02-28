# WhisperForge

## Structure
- `app.py` — Gradio web UI with two tabs: **Collect Data** and **Train**
- `train.py` — Whisper fine-tuning script (run standalone or via Train tab)
- `data_collators.py` — Custom data collator for speech seq2seq
- `userdata/` — All user data (gitignored, mounted as volume in Docker)
  - `labels/` — `.txt` files with sentences (one per line)
  - `dataset/` — `audio_files/` and `recorded_samples.csv` (train/val split happens at training time)

## Running
```bash
pip install -r requirements.txt
python app.py
```

## Key Details
- Audio: 16kHz mono WAV
- CSV format: `audio,text` with text in double quotes
- Train tab runs `train.py` as a subprocess
- All data paths are under `userdata/`
