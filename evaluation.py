import logging
import sys
import traceback
from datetime import datetime

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator, SimilarityFunction
from sentence_transformers.training_args import BatchSamplers

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

batch_size = 256  # The larger you select this, the better the results (usually). But it requires more GPU memory

matryoshka_dims = [768, 512, 256, 128, 64]

model = SentenceTransformer('AbderrahmanSkiredj1/Arabic_text_embedding_for_sts')

# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

model.save("output/matryoshkav4_fromv3_2700")
model.push_to_hub('AbderrahmanSkiredj1/Arabic_text_embedding_for_sts')


eval_dataset_name = "mteb/sts17-crosslingual-sts"
stsb_eval_dataset = load_dataset(eval_dataset_name, "ar-ar", split="test")
evaluators = []

for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=stsb_eval_dataset["sentence1"],
            sentences2=stsb_eval_dataset["sentence2"],
            scores=stsb_eval_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-dev-{dim}",
            truncate_dim=dim,
        )
    )

dev_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[0])


# 7. Evaluate the model performance on the STS Benchmark test dataset
#test_dataset = load_dataset(eval_dataset_name, split="test")
test_dataset = load_dataset(eval_dataset_name, "ar-ar", split="test")
evaluators = []
for dim in matryoshka_dims:
    evaluators.append(
        EmbeddingSimilarityEvaluator(
            sentences1=test_dataset["sentence1"],
            sentences2=test_dataset["sentence2"],
            scores=test_dataset["score"],
            main_similarity=SimilarityFunction.COSINE,
            name=f"sts-test-{dim}",
            truncate_dim=dim,
        )
    )
test_evaluator = SequentialEvaluator(evaluators)
test_evaluator(model)
