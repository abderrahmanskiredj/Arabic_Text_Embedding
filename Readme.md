# Arabic Text Embedding for Semantic Similarity

This repository contains the code and resources for the top-ranked Arabic Text Embedding model on the MTEB leaderboard, achieving a cosine Spearman correlation of 85.

## Model Overview

Our model leverages the AraBERTv02 pretrained model and is fine-tuned on a diverse and enriched dataset composed of five different sources. It excels in tasks requiring high semantic similarity, particularly in Arabic Retrieval-Augmented Generation (RAG) applications.

### Model Link
[Arabic Text Embedding for STS](https://huggingface.co/AbderrahmanSkiredj1/Arabic_text_embedding_for_sts)

### Model Description
- **Model Type:** Sentence Transformer
- **Base Model:** AraBERTv02
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 768 tokens
- **Similarity Function:** Cosine Similarity


## Training Dataset

The model was trained on the [ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative](https://huggingface.co/datasets/AbderrahmanSkiredj1/ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative) dataset, which is a concatenation of five datasets, augmented and enriched automatically.

## Performance

The model outperforms several commonly used multilingual models, including:
- multilingual-e5-large
- paraphrase-multilingual-mpnet-base
- LaBSE
- Cohere Embedding

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First, install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference:
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("AbderrahmanSkiredj1/Arabic_text_embedding_for_sts")

# Run inference
sentences = [
    'يتم إنتاج أمثلة جميلة من المينا، والسيراميك، والفخار في وفرة كبيرة، وغالبا ما تتبع موضوع سلتيكي.',
    'يتم إنتاج عدد كبير من العناصر ذات المواضيع السلتية.',
    'يتم إنتاج الفخار الصغير الذي له موضوع سلتيكي.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
from sentence_transformers.util import cos_sim
similarities = cos_sim(embeddings, embeddings)
print(similarities)
```

## Training Details

### Training Dataset

#### AbderrahmanSkiredj1/arabic_quora_duplicates_stsb_alue_holyquran_aranli_900k_anchor_positive_negative

* Dataset: [ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative](https://huggingface.co/datasets/AbderrahmanSkiredj1/ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative)
* Size: 853,827 training samples
* Columns: `anchor`, `positive`, and `negative`

### Evaluation Dataset

#### AbderrahmanSkiredj1/arabic_quora_duplicates_stsb_alue_holyquran_aranli_900k_anchor_positive_negative

* Dataset: [ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative](https://huggingface.co/datasets/AbderrahmanSkiredj1/ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative)
* Size: 11,584 evaluation samples
* Columns: `anchor`, `positive`, and `negative`

### Training Logs
| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0120 | 40   | 3.1459        |
| 0.0240 | 80   | 3.2058        |
| 0.0360 | 120  | 3.0837        |
| 0.0480 | 160  | 3.1024        |
| 0.0600 | 200  | 3.015         |
| 0.0719 | 240  | 3.1311        |
| 0.0839 | 280  | 3.1101        |
| 0.0959 | 320  | 3.1288        |
| 0.1079 | 360  | 3.045         |
| 0.1199 | 400  | 3.0488        |
| 0.1319 | 440  | 3.1001        |
| 0.1439 | 480  | 3.2334        |
| 0.1559 | 520  | 3.0581        |
| 0.1679 | 560  | 2.9821        |
| 0.1799 | 600  | 3.1733        |
| 0.1918 | 640  | 3.0658        |
| 0.2038 | 680  | 3.0721        |
| 0.2158 | 720  | 3.1647        |
| 0.2278 | 760  | 3.0326        |
| 0.2398 | 800  | 3.1014        |
| 0.2518 | 840  | 2.9365        |
| 0.2638 | 880  | 3.0642        |
| 0.2758 | 920  | 2.9864        |
| 0.2878 | 960  | 3.0939        |
| 0.2998 | 1000 | 3.0676        |
| 0.3118 | 1040 | 2.9717        |
| 0.3237 | 1080 | 2.9908        |
| 0.3357 | 1120 | 2.9506        |
| 0.3477 | 1160 | 2.907         |
| 0.3597 | 1200 | 3.0451        |
| 0.3717 | 1240 | 3.0002        |
| 0.3837 | 1280 | 2.8842        |
| 0.3957 | 1320 | 3.0697        |
| 0.4077 | 1360 | 2.8967        |
| 0.4197 | 1400 | 3.0008        |
| 0.4317 | 1440 | 3.0027        |
| 0.4436 | 1480 | 2.9229        |
| 0.4556 | 1520 | 2.9539        |
| 0.4676 | 1560 | 2.9415        |
| 0.4796 | 1600 | 2.9401        |
| 0.4916 | 1640 | 2.8498        |
| 0.5036 | 1680 | 2.9646        |
| 0.5156 | 1720 | 2.9231        |
| 0.5276 | 1760 | 2.942         |
| 0.5396 | 1800 | 2.8521        |
| 0.5516 | 1840 | 2.8362        |
| 0.5635 | 1880 | 2.8497        |
| 0.5755 | 1920 | 2.8867        |
| 0.5875 | 1960 | 2.9148        |
| 0.5995 | 2000 | 2.9343        |
| 0.6115 | 2040 | 2.8537        |
| 0.6235 | 2080 | 2.7989        |
| 0.6355 | 2120 | 2.8508        |
| 0.6475 | 2160 | 2.916         |
| 0.6595 | 2200 | 2.926         |
| 0.6715 | 2240 | 2.752         |
| 0.6835 | 2280 | 2

.7792        |
| 0.6954 | 2320 | 2.8381        |
| 0.7074 | 2360 | 2.7455        |
| 0.7194 | 2400 | 2.8953        |
| 0.7314 | 2440 | 2.8179        |
| 0.7434 | 2480 | 2.8471        |
| 0.7554 | 2520 | 2.7538        |
| 0.7674 | 2560 | 2.8271        |
| 0.7794 | 2600 | 2.8401        |
| 0.7914 | 2640 | 2.7402        |
| 0.8034 | 2680 | 2.6439        |

### Framework Versions
- Python: 3.10.14
- Sentence Transformers: 3.0.1
- Transformers: 4.39.3
- PyTorch: 2.2.2+cu121
- Accelerate: 0.29.1
- Datasets: 2.18.0
- Tokenizers: 0.15.2

## Reproducibility

The training code and evaluation scripts are provided to ensure reproducibility. You can find them in the `src` directory.

## Citation

If you use the Arabic Matryoshka Embeddings Model, please cite it as follows:

```bibtex
@software{skiredj2024,
  author       = {Abderrahman Skiredj},
  title        = {Arabic Text Embedding for Semantic Text Similarity},
  year         = 2024,
  url          = {https://huggingface.co/AbderrahmanSkiredj1/Arabic_text_embedding_for_sts},
  version      = {1.0.0},
}

## License

This project is licensed under the MIT License.


Happy coding! 🚀
