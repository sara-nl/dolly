#!/bin/bash

module load 2022 Python/3.10.4-GCCcore-11.3.0-bare

# Download PDM
curl -sSL https://raw.githubusercontent.com/pdm-project/pdm/main/install-pdm.py | python3 -

pdm install
