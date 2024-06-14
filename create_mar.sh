#!/bin/bash

mkdir ./model_store || echo "model_store directory already exists"

torch-model-archiver --model-name "sigmoid_output_model" \
--version 1.0 \
--serialized-file ./app/models/checkpoints/20240523_135239/epoch_0039.pt \
--handler ./app/handlers/sigmoid_output_handler.py \
--export-path ./model_store