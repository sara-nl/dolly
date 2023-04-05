import logging
from subprocess import run
from pathlib import Path


logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
)
logging.getLogger("py4j").setLevel(logging.WARNING)
logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

import os
from datetime import datetime
from training.utils_old import load_training_dataset, load_tokenizer

# COMMAND ----------

# Cache data and tokenizer locally before creating a bunch of deepspeed processes and make sure they succeeds.
load_training_dataset()
load_tokenizer()

# COMMAND ----------

timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
model_name = "dolly"
checkpoint_dir_name = f"{model_name}__{timestamp}"

root_path = os.getcwd()
deepspeed_config = os.path.join(root_path, "config/ds_z3_bf16_config.json")

dolly_training_dir_name = "dolly_training"

# Use the local training root path if it was provided.  Otherwise try to find a sensible default.
local_training_root = os.path.join(os.path.expanduser('~'), dolly_training_dir_name)

os.makedirs(local_training_root, exist_ok=True)

local_output_dir = os.path.join(local_training_root, checkpoint_dir_name)

tensorboard_display_dir = f"{local_output_dir}/runs"

print(f"Local Output Dir: {local_output_dir}")
print(f"Tensorboard Display Dir: {tensorboard_display_dir}")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


num_gpus = 1


current_filepath = str(Path(os.path.realpath(__file__)).parent)
run(" ".join([
    f"HF_DATASETS_CACHE={current_filepath}/.cache/huggingface/datasets/",
    "deepspeed",
    "--module", "training.trainer",
    "--deepspeed", f"{deepspeed_config}",
    "--epochs", "1",
    "--local-output-dir", f"{local_output_dir}",
    "--per-device-train-batch-size", "8",
    "--per-device-eval-batch-size", "8",
    "--lr", "1e-5",
]), shell=True)

from training.generate import generate_response, load_model_tokenizer_for_generate

model, tokenizer = load_model_tokenizer_for_generate(local_output_dir)

# COMMAND ----------

# Examples from https://www.databricks.com/blog/2023/03/24/hello-dolly-democratizing-magic-chatgpt-open-models.html
instructions = [
    "Write a love letter to Edgar Allan Poe.",
    "Write a tweet announcing Dolly, a large language model from Databricks.",
    "I'm selling my Nikon D-750, write a short blurb for my ad.",
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
]

# Use the model to generate responses for each of the instructions above.
for instruction in instructions:
    response = generate_response(instruction, model=model, tokenizer=tokenizer)
    if response:
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")

