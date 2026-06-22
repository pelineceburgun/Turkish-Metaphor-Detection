
import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel, Value
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    df = pd.read_parquet(args.data)
    df["n_metaphor"] = df["labels"].apply(sum)
    df["label_sent"] = (df["n_metaphor"] > 0).astype(int)
    text_col = "sentence_tr" if "sentence_tr" in df.columns else "sentence"

    hf   = Dataset.from_pandas(df[[text_col, "label_sent"]].rename(columns={"label_sent": "labels"}))
    hf   = hf.cast_column("labels", ClassLabel(names=["LITERAL", "METAPHOR"]))
    test = hf.train_test_split(test_size=0.2, seed=42, stratify_by_column="labels")["test"]
    test = test.cast_column("labels", Value("int64"))
    test = test.map(
        lambda b: tokenizer(b[text_col], truncation=True, max_length=128, padding="max_length"),
        batched=True, remove_columns=[text_col]
    )

    trainer    = Trainer(model=model)
    preds_out  = trainer.predict(test)
    pred_labels = np.argmax(preds_out.predictions, axis=-1)

    print(classification_report(
        preds_out.label_ids, pred_labels,
        target_names=["LITERAL", "METAPHOR"], digits=4
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data",       default="data/vua20_tr_full.parquet")
    args = parser.parse_args()
    main(args)


print("evaluate.py kaydedildi.")
print("shap_analysis.py kaydedildi.")
