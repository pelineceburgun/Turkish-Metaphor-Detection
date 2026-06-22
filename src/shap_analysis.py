
import os
import argparse
import shap
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

matplotlib.rcParams["font.family"] = "DejaVu Sans"


def main(args):
    os.makedirs("figures", exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model     = AutoModelForSequenceClassification.from_pretrained(args.model_path)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
    )

    df = pd.read_parquet(args.data)
    df["n_metaphor"] = df["labels"].apply(sum)
    df["label_sent"] = (df["n_metaphor"] > 0).astype(int)
    text_col = "sentence_tr" if "sentence_tr" in df.columns else "sentence"

    metaphor_sents = df[df["label_sent"] == 1][text_col].tolist()[:args.n_samples]
    literal_sents  = df[df["label_sent"] == 0][text_col].tolist()[:args.n_samples]
    test_sentences = metaphor_sents + literal_sents

    explainer   = shap.Explainer(pipe)
    shap_values = explainer(test_sentences)

    records = []
    for i, sentence in enumerate(test_sentences):
        tokens = shap_values[i].data
        values = shap_values[i].values[:, 1]  # METAPHOR class
        label  = "METAPHOR" if i < args.n_samples else "LITERAL"
        for tok, val in zip(tokens, values):
            records.append({"token": tok, "shap_value": val, "label": label})

    shap_df   = pd.DataFrame(records)
    shap_means = shap_df.groupby("token")["shap_value"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    top_pos = shap_means.sort_values(ascending=False).head(15)
    axes[0].barh(top_pos.index[::-1], top_pos.values[::-1], color="#e74c3c", alpha=0.8)
    axes[0].set_title("Tokens increasing METAPHOR probability", fontsize=12)
    axes[0].set_xlabel("Mean SHAP value")
    axes[0].axvline(0, color="black", linewidth=0.8)

    top_neg = shap_means.sort_values().head(15)
    axes[1].barh(top_neg.index[::-1], top_neg.values[::-1], color="#3498db", alpha=0.8)
    axes[1].set_title("Tokens increasing LITERAL probability", fontsize=12)
    axes[1].set_xlabel("Mean SHAP value")
    axes[1].axvline(0, color="black", linewidth=0.8)

    plt.suptitle("BERTurk Metaphor Detection — SHAP Token Importance", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/shap_analysis.png", dpi=150, bbox_inches="tight")
    print("Saved to figures/shap_analysis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data",       default="data/vua20_tr_full.parquet")
    parser.add_argument("--n_samples",  type=int, default=25)
    args = parser.parse_args()
    main(args)
