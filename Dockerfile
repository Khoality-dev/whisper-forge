FROM node:20-slim AS frontend
WORKDIR /frontend
COPY frontend/ .
RUN npm install && npm run build

FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py train.py predict.py data_collators.py ./
COPY --from=frontend /static ./static/
RUN mkdir -p userdata/labels userdata/dataset/audio_files
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
