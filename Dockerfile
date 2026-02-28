FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py data_collators.py ./

RUN mkdir -p userdata/labels userdata/dataset/audio_files

EXPOSE 7860

CMD ["python", "app.py"]
