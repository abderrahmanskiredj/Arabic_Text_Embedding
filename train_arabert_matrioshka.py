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

#model_name = sys.argv[1] if len(sys.argv) > 1 else "distilroberta-base"
model_name = "aubmindlab/bert-base-arabertv02"
model_name= "AbderrahmanSkiredj1/arabic_text_embedding_sts_arabertv02_arabicnlitriplet"

batch_size = 64  # The larger you select this, the better the results (usually). But it requires more GPU memory
num_train_epochs = 10
matryoshka_dims = [768, 512, 256, 128, 64]

# Save path of the model
output_dir = f"output/matryoshka_arabert_{model_name.replace('/', '-')}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# 1. Here we define our SentenceTransformer model. If not already a Sentence Transformer model, it will automatically
# create one with "mean" pooling.
model = SentenceTransformer(model_name)
# If we want, we can limit the maximum sequence length for the model
# model.max_seq_length = 75
logging.info(model)

# 2. Load the AllNLI dataset: https://huggingface.co/datasets/sentence-transformers/all-nli

dataset_name = "AbderrahmanSkiredj1/ArabicQuoraDuplicates_stsb_Alue_holyquran_aranli_900k_anchor_positive_negative"

'''train_dataset = load_dataset(dataset_name, "triplet", split="train")
eval_dataset = load_dataset(dataset_name, "triplet", split="dev")'''
train_dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="validation")
logging.info(train_dataset)

# If you wish, you can limit the number of training samples
# train_dataset = train_dataset.select(range(5000))

# 3. Define our training loss
inner_train_loss = losses.MultipleNegativesRankingLoss(model)
train_loss = losses.MatryoshkaLoss(model, inner_train_loss, matryoshka_dims=matryoshka_dims)

# 4. Define an evaluator for use during training. This is useful to keep track of alongside the evaluation loss.


eval_dataset_name = "Omartificial-Intelligence-Space/Arabic-stsb"
stsb_eval_dataset = load_dataset(eval_dataset_name, split="validation")
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

# 5. Define the training arguments
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_ratio=0.1,
    fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
    bf16=False,  # Set to True if you have a GPU that supports BF16
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    # Optional tracking/debugging parameters:
    do_eval=True,
    #eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=300,
    #save_total_limit=2,
    logging_steps=40,
    run_name="matryoshka-nli",  # Will be used in W&B if `wandb` is installed
    gradient_accumulation_steps=1,
    weight_decay=0,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    max_grad_norm=1.0,
    lr_scheduler_type="linear",
    seed=42,
    learning_rate=1e-6,
)
'''gradient_accumulation_steps=1,
weight_decay=0,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-8,
max_grad_norm=1.0,
lr_scheduler_type="linear",
seed=42'''
# 6. Create the trainer & start training
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=dev_evaluator,
)
trainer.train()

# 7. Evaluate the model performance on the STS Benchmark test dataset
test_dataset = load_dataset(eval_dataset_name, split="test")
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

# 8. Save the trained & evaluated model locally
final_output_dir = f"{output_dir}/final"
model.save(final_output_dir)
