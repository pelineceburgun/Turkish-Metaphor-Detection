
data_prep = '''"""
data_preparation.py
-------------------
Downloads VUA20 from HuggingFace, groups tokens into sentences,
translates to Turkish via DeepL, and saves the dataset.

Usage:
    export DEEPL_API_KEY="your-key-here"
    python src/data_preparation.py
"""

import os
import time
import pandas as pd
from datasets import load_dataset

DEEPL_API_KEY = os.environ.get("DEEPL_API_KEY", "your-key-here")
OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "data/")
KOTA          = 480_000  # DeepL free tier: 500k chars/month


def load_vua20():
    """Load VUA20 from HuggingFace and return train/test DataFrames."""
    vua20    = load_dataset("CreativeLang/vua20_metaphor")
    df_train = pd.DataFrame(vua20["train"])
    df_test  = pd.DataFrame(vua20["test"])
    print(f"VUA20 loaded: {len(df_train)} train / {len(df_test)} test tokens")
    return df_train, df_test


def group_by_sentence(df):
    """Group token-level rows into sentence-level records."""
    grouped = []
    for idx, group in df.groupby("index", sort=False):
        group      = group.sort_values("w_index").reset_index(drop=True)
        sentence   = group["sentence"].iloc[0]
        labels     = group["label"].tolist()
        pos_tags   = group["POS"].tolist()
        w_indices  = group["w_index"].tolist()
        all_tokens = sentence.split()

        if len(w_indices) == 0 or max(w_indices) >= len(all_tokens):
            continue

        tokens = [all_tokens[i] for i in w_indices]
        grouped.append({
            "sentence":   sentence,
            "tokens":     tokens,
            "labels":     labels,
            "pos":        pos_tags,
            "w_indices":  w_indices,
            "index":      idx,
            "n_tokens":   len(tokens),
            "n_metaphor": sum(labels),
        })
    return pd.DataFrame(grouped)


def select_within_quota(train_grouped, kota=KOTA):
    """Select balanced subset that fits within DeepL character quota."""
    has_met = train_grouped[train_grouped["n_metaphor"] > 0]
    no_met  = train_grouped[train_grouped["n_metaphor"] == 0]
    n       = min(len(has_met), len(no_met))

    balanced = pd.concat([
        has_met.sample(n, random_state=42),
        no_met.sample(n, random_state=42),
    ]).reset_index(drop=True)
    balanced["sentence_len"] = balanced["sentence"].str.len()

    # Metaphor-first priority: 60% of quota
    met_quota = int(kota * 0.6)
    has_sorted = balanced[balanced["n_metaphor"] > 0].sort_values("sentence_len")
    has_sorted["cumsum"] = has_sorted["sentence_len"].cumsum()
    met_within = has_sorted[has_sorted["cumsum"] <= met_quota]

    lit_quota = kota - met_within["sentence_len"].sum()
    no_sorted = balanced[balanced["n_metaphor"] == 0].sort_values("sentence_len")
    no_sorted["cumsum"] = no_sorted["sentence_len"].cumsum()
    lit_within = no_sorted[no_sorted["cumsum"] <= lit_quota]

    final = pd.concat([met_within, lit_within]).reset_index(drop=True)
    print(f"Selected {len(final)} sentences ({final['sentence_len'].sum():,} chars)")
    return final


def translate_to_turkish(df, api_key=DEEPL_API_KEY, batch_size=50):
    """Translate English sentences to Turkish using DeepL."""
    import deepl
    translator = deepl.Translator(api_key)
    sentences  = df["sentence"].tolist()
    translated = []

    for i in range(0, len(sentences), batch_size):
        batch   = sentences[i:i + batch_size]
        results = translator.translate_text(batch, target_lang="TR")
        translated.extend([r.text for r in results])
        time.sleep(0.5)
        if i % 500 == 0:
            print(f"  {len(translated)}/{len(sentences)} translated")

    df["sentence_tr"] = translated
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load and group
    df_train, df_test = load_vua20()
    train_grouped = group_by_sentence(df_train)
    test_grouped  = group_by_sentence(df_test)
    print(f"Grouped: {len(train_grouped)} train / {len(test_grouped)} test sentences")

    # Sentence-level labels
    train_grouped["label_sent"] = (train_grouped["n_metaphor"] > 0).astype(int)
    test_grouped["label_sent"]  = (test_grouped["n_metaphor"]  > 0).astype(int)

    # Save English grouped data
    train_grouped.to_parquet(os.path.join(OUTPUT_DIR, "vua20_en_train.parquet"))
    test_grouped.to_parquet(os.path.join(OUTPUT_DIR, "vua20_en_test.parquet"))

    # Select quota-safe subset and translate
    to_translate = select_within_quota(train_grouped)
    print(f"Translating {len(to_translate)} sentences to Turkish...")
    translated = translate_to_turkish(to_translate)
    translated["label_sent"] = (translated["n_metaphor"] > 0).astype(int)
    translated.to_parquet(os.path.join(OUTPUT_DIR, "vua20_tr_full.parquet"))
    print(f"Saved to {OUTPUT_DIR}vua20_tr_full.parquet")


if __name__ == "__main__":
    main()
'''

with open("/content/drive/MyDrive/metaphor_project/src/data_preparation.py", "w") as f:
    f.write(data_prep)
print("data_preparation.py kaydedildi.")
