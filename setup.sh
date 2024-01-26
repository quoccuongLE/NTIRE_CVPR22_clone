#!/bin/bash
VENV_DIR=${1:-".venv/NTIRE"}
conda env create -f conda.yaml --prefix $VENV_DIR
