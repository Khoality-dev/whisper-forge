# WhisperForge

## Structure
- `app.py` — FastAPI backend (API endpoints + serves React static build)
- `train.py` — Whisper fine-tuning script (run standalone or via Train tab)
- `data_collators.py` — Custom data collator for speech seq2seq
- `frontend/` — React (Vite) source code
  - `src/App.jsx` — Main app with tab navigation
  - `src/components/CollectData.jsx` — Recording UI (browser MediaRecorder)
  - `src/components/Train.jsx` — Training config + log polling
  - `src/api.js` — API client helpers
- `static/` — Vite build output (gitignored, served by FastAPI at `/`)
- `userdata/` — All user data (gitignored, mounted as volume in Docker)
  - `labels/` — `.txt` files with sentences (one per line)
  - `dataset/` — `audio_files/` and `recorded_samples.csv` (train/val split happens at training time)

## Running
```bash
# Build frontend
cd frontend && npm install && npm run build && cd ..

# Run backend
pip install -r requirements.txt
python app.py
```

## API Endpoints
- `GET /api/sentences/current` — current sentence + progress
- `POST /api/sentences/skip` — skip to next sentence
- `POST /api/recordings` — save audio recording (multipart: audio file + sentence text)
- `POST /api/train/start` — start training subprocess (JSON config body)
- `POST /api/train/stop` — stop training subprocess
- `GET /api/train/status` — training status + log output

## Key Details
- Audio: recorded in browser via MediaRecorder, uploaded as WAV
- CSV format: `audio,text` with text in double quotes
- Train tab runs `train.py` as a subprocess
- All data paths are under `userdata/`
- Frontend dev server proxies `/api` to `localhost:7860`
