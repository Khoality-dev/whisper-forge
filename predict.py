import argparse
import csv
import json
import os
import random
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


def scan_samples(dataset_dirs):
    """Read labels.csv + recorded_samples.csv from each dataset dir, return joined list."""
    samples = []
    for ds_dir in dataset_dirs:
        labels_path = os.path.join(ds_dir, "labels.csv")
        recordings_path = os.path.join(ds_dir, "recorded_samples.csv")
        if not os.path.exists(labels_path) or not os.path.exists(recordings_path):
            continue

        with open(labels_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            id_to_text = {row["id"]: row["text"] for row in reader}

        with open(recordings_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
            for row in reader:
                text = id_to_text.get(row["label_id"])
                if text is not None:
                    samples.append({"audio": row["audio"], "text": text})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="userdata/models/v1")
    parser.add_argument("--dataset_dirs", type=str, nargs="+", required=True,
                        help="Dataset directories (each with labels.csv, recorded_samples.csv, audio_files/)")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    samples = scan_samples(args.dataset_dirs)

    if not samples:
        print(json.dumps([]))
        return

    selected = random.sample(samples, min(args.n, len(samples)))

    # Load model
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()

    results = []
    for s in selected:
        audio, sr = librosa.load(s["audio"], sr=16000)
        inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            predicted_ids = model.generate(inputs.input_features)
        predicted = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        results.append({
            "audio": s["audio"],
            "expected": s["text"],
            "predicted": predicted.strip(),
        })

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
