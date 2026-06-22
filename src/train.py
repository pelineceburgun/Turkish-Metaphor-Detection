
import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel, Value
from sklearn.metrics import f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_MAP = {
    "bertturk": "dbmdz/bert-base-turkish-128k-uncased",
    "mbert":    "bert-base-multilingual-cased",
}

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "results/")


def load_data(parquet_path):
    df = pd.read_parquet(parquet_path)
    df["n_metaphor"] = df["labels"].apply(sum)
    df["label_sent"] = (df["n_metaphor"] > 0).astype(int)

    text_col = "sentence_tr" if "sentence_tr" in df.columns else "sentence"
    return df, text_col


def build_datasets(df, text_col, tokenizer):
    hf = Dataset.from_pandas(
        df[[text_col, "label_sent"]].rename(columns={"label_sent": "labels"})
    )
    hf = hf.cast_column("labels", ClassLabel(names=["LITERAL", "METAPHOR"]))

    splits     = hf.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")
    train_val  = splits["train"].train_test_split(test_size=0.1, seed=42, stratify_by_column="labels")

    def tokenize(batch):
        return tokenizer(batch[text_col], truncation=True, max_length=128, padding="max_length")

    train = train_val["train"].cast_column("labels", Value("int64"))
    val   = train_val["test"].cast_column("labels",  Value("int64"))
    test  = splits["test"].cast_column("labels",     Value("int64"))

    train = train.map(tokenize, batched=True, remove_columns=[text_col])
    val   = val.map(tokenize,   batched=True, remove_columns=[text_col])
    test  = test.map(tokenize,  batched=True, remove_columns=[text_col])

    return train, val, test


def compute_metrics(pred):
    logits, labels = pred
    preds = np.argmax(logits, axis=-1)
    return {
        "f1_macro":    f1_score(labels, preds, average="macro"),
        "f1_metaphor": f1_score(labels, preds, pos_label=1, average="binary"),
    }


def main(args):
    model_name = MODEL_MAP[args.model]
    print(f"Model: {model_name}")
    print(f"Data:  {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    df, text_col = load_data(args.data)
    train, val, test = build_datasets(df, text_col, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "LITERAL", 1: "METAPHOR"},
        label2id={"LITERAL": 0, "METAPHOR": 1},
    )

    run_name  = f"{args.model}_{os.path.basename(args.data).replace('.parquet','')}"
    train_args = TrainingArguments(
        output_dir                  = os.path.join(OUTPUT_DIR, f"checkpoints_{run_name}"),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = 16,
        per_device_eval_batch_size  = 32,
        learning_rate               = 2e-5,
        weight_decay                = 0.01,
        warmup_steps                = 200,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_metaphor",
        logging_steps               = 100,
        fp16                        = True,
        report_to                   = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = train_args,
        train_dataset   = train,
        eval_dataset    = val,
        compute_metrics = compute_metrics,
    )

    trainer.train()

    # Evaluate on test
    from sklearn.metrics import classification_report
    preds_out   = trainer.predict(test)
    pred_labels = np.argmax(preds_out.predictions, axis=-1)
    print(f"\\n=== Test Results: {run_name} ===")
    print(classification_report(
        preds_out.label_ids, pred_labels,
        target_names=["LITERAL", "METAPHOR"], digits=4
    ))

    # Save model
    save_path = os.path.join(OUTPUT_DIR, f"model_{run_name}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  choices=["bertturk", "mbert"], default="bertturk")
    parser.add_argument("--data",   default="data/vua20_tr_full.parquet")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)
