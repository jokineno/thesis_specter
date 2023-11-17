#!/bin/bash

echo "Starting training script..."

./scripts/run-exp-simple.sh \
  -c experiment_configs/simple.jsonnet \
  -s model-output \
  --num-epochs 2 \
  --batch-size 4 \
  --train-path data/preprocessed2/data-train.p \
  --dev-path data/preprocessed2/data-val.p \
  --num-train-instances 60 \
  --cuda-device -1

echo "DONE. Training finished.."