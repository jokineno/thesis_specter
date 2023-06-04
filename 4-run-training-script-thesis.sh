#!/bin/bash

echo "Starting training script..."

echo "Creating training files (thesis version).. "


export USE_HF_SPECIAL_TOKENS=True

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
  --bert-vocab thesis_data/finnish_bert_cased/vocab.txt \
  --bert-weights thesis_data/finnish_bert_cased/finnish_bert.tar.gz \
  --vocab thesis_data/finnish_bert_cased/vocabulary \
  --num-train-instances 35 \
  --cuda-device -1
else

  echo "Using DEMO samples for training.. ====[NOT THE REAL FULL DATA]===="
  OUTPUT_PATH="thesis-model-output"
  rm -r $OUTPUT_PATH
  echo "Removed ${OUTPUT_PATH}"


  ./scripts/run-exp-simple.sh \
  -c experiment_configs/simple_thesis.jsonnet \
  -s thesis-model-output/ \
  --num-epochs 2 \
  --batch-size 4 \
  --train-path thesis_data/preprocessed_demo/data-train.p \
  --dev-path thesis_data/preprocessed_demo/data-val.p \
  --bert-vocab thesis_data/finnish_bert_cased/vocab.txt \
  --bert-weights thesis_data/finnish_bert_cased/finnish_bert.tar.gz \
  --vocab thesis_data/finnish_bert_cased/vocabulary \
  --num-train-instances 35 \
  --cuda-device -1
fi

echo "DONE. Training finished.."
