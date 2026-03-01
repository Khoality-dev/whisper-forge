import argparse
import csv
import json
import random
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="userdata/outputs")
    parser.add_argument("--labels", type=str, default="userdata/dataset/labels.csv")
    parser.add_argument("--recordings", type=str, default="userdata/dataset/recorded_samples.csv")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    # Join labels + recordings
    with open(args.labels, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        id_to_text = {row["id"]: row["text"] for row in reader}

    with open(args.recordings, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|", quoting=csv.QUOTE_NONE)
        samples = []
        for row in reader:
            text = id_to_text.get(row["label_id"])
            if text is not None:
                samples.append({"audio": row["audio"], "text": text})

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
