# Turkish Metaphor Detection with BERTurk

Sentence-level metaphor detection for Turkish using BERTurk, trained on a machine-translated version of the VUA20 Metaphor Corpus.

## Key Results

| Model | Data | METAPHOR F1 | Macro F1 |
|-------|------|-------------|----------|
| BERTurk | EN VUA20 (zero-shot TR) | 0.00 | 0.43 |
| BERTurk | EN+TR few-shot (300 sent) | 0.40 | 0.60 |
| mBERT | TR VUA20-TR (10k sent) | 0.73 | 0.74 |
| **BERTurk** | **TR VUA20-TR (10k sent)** | **0.77** | **0.79** |

## Key Findings

1. **Zero-shot cross-lingual transfer fails completely** for Turkish metaphor detection (F1=0.00), confirming the need for Turkish-specific training data.
2. **Language-specific model outperforms multilingual baseline**: BERTurk (F1=0.77) > mBERT (F1=0.73) on Turkish data.
3. **Translation erodes metaphorical load**: False negatives concentrate in physical-action metaphors that appear literal after Turkish translation.
4. **SHAP analysis reveals asymmetric learning**: BERTurk identifies metaphors through lexical content tokens, but identifies literal sentences through morphological suffixes.

## Data

This project uses [VUA20 Metaphor Corpus](https://huggingface.co/datasets/CreativeLang/vua20_metaphor) translated to Turkish via DeepL API.

Raw data is not included in this repository due to licensing. To reproduce:
1. Download VUA20 from HuggingFace: `CreativeLang/vua20_metaphor`
2. Run `notebooks/01_data_preparation.ipynb` to generate the Turkish translation (requires DeepL API key)


## Notebooks

| Notebook | Description |
|----------|-------------|
| 01_data_preparation | VUA20 loading, grouping, DeepL translation |
| 02_baseline_training | BERTurk/mBERT token-level and sentence-level baselines |
| 03_turkish_finetuning | Zero-shot, few-shot, full Turkish fine-tuning |
| 04_analysis_shap | SHAP token importance + error analysis |

## Model

- **BERTurk**: `dbmdz/bert-base-turkish-128k-uncased`
- **mBERT**: `bert-base-multilingual-cased`
- Task: Binary sentence-level classification (METAPHOR / LITERAL)
- Training data: 10,044 Turkish sentences (VUA20 translated)

## Theoretical Background

Built on Lakoff & Johnson's Conceptual Metaphor Theory (1980). Annotation follows MIP/MIPVU protocol (Steen et al., 2010).

## References

- Lakoff, G., & Johnson, M. (1980). *Metaphors We Live By*. University of Chicago Press.
- Steen, G. et al. (2010). *A Method for Linguistic Metaphor Identification*. John Benjamins.
- Schweter, S. (2020). BERTurk — BERT Language Models for Turkish. Zenodo.
- Leong, C.W. et al. (2020). A Report on the 2020 VUA and TOEFL Metaphor Detection Shared Task. *ACL*.
