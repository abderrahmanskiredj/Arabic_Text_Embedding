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

#model = SentenceTransformer("output/matryoshka_nli_aubmindlab-bert-base-arabertv02-2024-07-06_19-10-50/checkpoint-10900")
#model = SentenceTransformer("output/matryoshka_nli_v2_aubmindlab-bert-base-arabertv02-2024-07-06_19-42-31/checkpoint-1300")
#model = SentenceTransformer('output/matryoshka_nli_v3fromv1_output-matryoshkav1_11000-2024-07-06_22-30-55/checkpoint-1200')
model = SentenceTransformer('output/matryoshka_nli_v4fromv3_output-matryoshka_nli_v3fromv1_output-matryoshkav1_11000-2024-07-06_22-30-55-checkpoint-1200-2024-07-07_12-56-26/checkpoint-2700')
#2700
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

model.save("output/matryoshkav4_fromv3_2700")
model.push_to_hub('AbderrahmanSkiredj1/Arabic_text_embedding_for_sts')

eval_dataset_name = "sentence-transformers/stsb"
eval_dataset_name = "Omartificial-Intelligence-Space/Arabic-stsb"
eval_dataset_name = "mteb/sts17-crosslingual-sts"
#stsb_eval_dataset = load_dataset(eval_dataset_name, split="validation")
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

# 9. (Optional) save the model to the Hugging Face Hub!
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
#model_name = model_name if "/" not in model_name else model_name.split("/")[-1]
'''
try:
    model.push_to_hub(f"{model_name}-nli-matryoshka")
except Exception:
    logging.error(
        f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
        f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
        f"and saving it using `model.push_to_hub('{model_name}-nli-matryoshka')`."
    )
'''