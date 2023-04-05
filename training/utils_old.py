import logging
from functools import partial
from typing import Union, Any

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .consts import (
    DEFAULT_SEED, DEFAULT_TRAINING_DATASET, END_KEY, DEFAULT_INPUT_MODEL, INSTRUCTION_KEY,
    RESPONSE_KEY_NL
)

logger = logging.getLogger(__name__)


def preprocess_batch(batch: dict[str, list], tokenizer: PreTrainedTokenizerBase, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.
        seed: (int): seed

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset()

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["instruction", "input", "output", "text"],
    )

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [END_KEY, INSTRUCTION_KEY, RESPONSE_KEY_NL]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def load_training_dataset(training_data_id: str = DEFAULT_TRAINING_DATASET, split: str = "train") -> Dataset:
    logger.info(f"Loading {training_data_id} dataset")
    dataset: Dataset = load_dataset(training_data_id)[split]
    logger.info("Found %d rows", dataset.num_rows)

    # Remove empty responses
    response_key_stripped = RESPONSE_KEY_NL.strip()
    dataset = dataset.filter(lambda rec: not rec["text"].strip().endswith(response_key_stripped))

    def _func(rec):
        rec["text"] += f"\n\n{END_KEY}"
        return rec

    dataset = dataset.map(_func)

    return dataset

class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        response_token_ids = self.tokenizer.encode(RESPONSE_KEY_NL)

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_token_ids_start_idx = None
            for idx in np.where(batch["labels"][i] == response_token_ids[0])[0]:
                response_token_ids_start_idx = idx
                break

            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Could not find response key {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


