# WhisperForge

## Structure
- `app.py` — FastAPI backend (API endpoints + serves React static build)
- `train.py` — Whisper fine-tuning script (run standalone or via Train tab)
- `predict.py` — Inference script for sample predictions
- `data_collators.py` — Custom data collator for speech seq2seq
- `frontend/` — React (Vite) source code
  - `src/App.jsx` — Main app with tab navigation
  - `src/components/Dataset.jsx` — Dataset list + sample management
  - `src/components/Collect.jsx` — Recording UI (browser MediaRecorder)
  - `src/components/Train.jsx` — Model version config + training + log polling
  - `src/api.js` — API client helpers
- `static/` — Vite build output (gitignored, served by FastAPI at `/`)
- `userdata/` — All user data (gitignored, mounted as volume in Docker)
  - `datasets/` — Named dataset directories, each containing `*.wav` + `*.txt` pairs
  - `models/` — Named model versions, each with `config.json` + trained model files

## Running
```bash
# Build frontend
cd frontend && npm install && npm run build && cd ..

# Run backend
pip install -r requirements.txt
python app.py
```

## API Endpoints

### Datasets
- `GET /api/datasets` — List all datasets (name + sample count)
- `POST /api/datasets` — Create dataset `{"name": "..."}`
- `DELETE /api/datasets/{name}` — Delete dataset + all files
- `GET /api/datasets/{name}` — List samples in dataset
- `POST /api/datasets/{name}/recordings` — Save recording (multipart: audio + text)
- `DELETE /api/datasets/{name}/recordings/{sample}` — Delete wav+txt pair
- `POST /api/datasets/{name}/upload` — Upload wav+txt files or zip
- `GET /api/audio/{dataset}/{filename}` — Serve audio file

### Models
- `GET /api/models` — List all model versions (name + status)
- `POST /api/models` — Create version `{"name": "..."}`
- `DELETE /api/models/{name}` — Delete version + files
- `GET /api/models/{name}/config` — Get version config
- `PUT /api/models/{name}/config` — Save version config
- `POST /api/models/{name}/train` — Start training this version
- `POST /api/models/{name}/stop` — Stop training
- `GET /api/models/{name}/status` — Training status + log
- `GET /api/models/{name}/download` — Download trained model zip
- `POST /api/models/{name}/predict` — Run test predictions

## Key Details
- Audio: recorded in browser via MediaRecorder, uploaded as WAV
- Dataset format: flat directory of `name.wav` + `name.txt` pairs
- Each model version stores `config.json` with selected datasets + hyperparameters
- Train tab runs `train.py` as a subprocess with `--dataset_dirs`
- Multiple model versions can exist independently with separate configs/outputs
- All data paths are under `userdata/`
- Frontend dev server proxies `/api` to `localhost:7860`
