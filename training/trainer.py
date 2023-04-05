# Copyright 2023 Databricks, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import click
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
)

from training.utils_old import get_model_tokenizer, preprocess_dataset, DataCollatorForCompletionOnlyLM
from .consts import (
    DEFAULT_SEED,
)

logger = logging.getLogger(__name__)



def train(
    local_output_dir,
    dbfs_output_dir,
    epochs,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    lr,
    seed,
    deepspeed,
    gradient_checkpointing,
    local_rank,
    bf16,
    test_size=1000,
):
    set_seed(seed)

    model, tokenizer = get_model_tokenizer(gradient_checkpointing=gradient_checkpointing)

    # Use the same max length that the model supports.  Try a couple different keys in case a different
    # model is used.  The default model uses n_positions.  If no config settings can be found just default
    # to 1024 as this is probably supported by most models.
    conf = model.config
    max_length: int = getattr(conf, "n_positions", getattr(conf, "seq_lenth", 1024))

    processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed)

    split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    if not dbfs_output_dir:
        logger.warning("Will NOT save to DBFS")

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="tensorboard",
        disable_tqdm=True,
        remove_unused_columns=False,
        local_rank=local_rank,
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=8, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=8, help="Batch size to use for evaluation.")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
