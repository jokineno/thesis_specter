#!/bin/bash

echo "Starting training script..."

echo "Creating training files (thesis version).. "

if [[ ! $* == *--demo* ]]
then

  echo "Using all the samples for training.. ====[THIS IS THE FULL TRAINING]===="
  ./scripts/run-exp-simple.sh \
  -c experiment_configs/simple_thesis.jsonnet \
  -s thesis-model-output/ \
  --num-epochs 2 \
  --batch-size 4 \
  --train-path thesis_data/preprocessed/data-train.p \
  --dev-path thesis_data/preprocessed/data-val.p \
  --num-train-instances 60 \
  --cuda-device -1
else

  echo "Using demo samples for training.. ====[NOT THE REAL FULL DATA]===="
  ./scripts/run-exp-simple.sh \
  -c experiment_configs/simple_thesis.jsonnet \
  -s thesis-model-output/ \
  --num-epochs 2 \
  --batch-size 4 \
  --train-path thesis_data/preprocessed_demo/data-train.p \
  --dev-path thesis_data/preprocessed_demo/data-val.p \
  --num-train-instances 60 \
  --cuda-device -1
fi

echo "DONE. Training finished.."