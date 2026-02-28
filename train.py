import argparse
import numpy as np
from datasets import load_dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from data_collators import DataCollatorSpeechSeq2Seq
import evaluate
from jiwer import wer

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper base model")
    parser.add_argument("--csv", type=str, default="userdata/dataset/recorded_samples.csv",
                        help="Path to the recorded samples CSV")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--output_dir", type=str, default="whisper-finetuned",
                        help="Where to save checkpoints and the final model")
    parser.add_argument("--lang", type=str, default="en",
                        help="Language code (e.g., 'en', 'de') for decoder start token")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fp16", action="store_true",
                        help="Enable mixed precision training (fp16)")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model and processor
    model_id = f"openai/whisper-base"
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language='English', task='transcribe')
    
    processor = WhisperProcessor.from_pretrained(model_id, language='English', task='transcribe')
    
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    model.generation_config.task = 'transcribe'
    
    model.generation_config.forced_decoder_ids = None


    # Load dataset and split into train/val
    ds = load_dataset("csv", data_files=args.csv, split="train")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.train_test_split(test_size=args.val_split, seed=42)
    ds["validation"] = ds.pop("test")

    # Preprocessing function
    def preprocess(batch):
        audio = batch["audio"]["array"]
        inputs = processor.feature_extractor(
            audio, sampling_rate=16_000, return_tensors="np"
        )
        batch["input_features"] = inputs.input_features[0]
        labels = processor.tokenizer(
            batch["text"], truncation=True, return_tensors="np"
        ).input_ids[0]
        batch["labels"] = labels
        return batch

    # Apply preprocessing
    ds = ds.map(
        preprocess,
        remove_columns=ds["train"].column_names,
        num_proc=4
    )

    # Data collator for dynamic padding
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Metric computation
    metric = evaluate.load('wer')
    
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
    
        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    
        return {'wer': wer}

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        predict_with_generate=True,
        logging_steps=args.logging_steps,
        eval_strategy ="steps",
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        fp16=args.fp16,
        report_to=["tensorboard"],
        metric_for_best_model="wer",
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    # Train and save
    trainer.train()
    trainer.save_model(args.output_dir)

    # Final evaluation
    metrics = trainer.evaluate()
    print(metrics)


if __name__ == "__main__":
    main()
