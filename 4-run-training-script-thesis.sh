#!/bin/bash

set -x

echo "[*] Starting training script..."

echo "[*] Creating training files (thesis version).. "


export USE_HF_SPECIAL_TOKENS=True

if [[ ! $* == *--demo* ]]
then

  echo "[*] Using all the samples for training.. ====[THIS IS THE FULL TRAINING]===="

  OUTPUT_PATH="thesis-model-output"
  rm -r $OUTPUT_PATH
  echo "[*] Removed data with rm -r  ${OUTPUT_PATH}"

  echo "[*] Make sure you have a correct number or training instances"
  echo "[*] Reading number of training instances from thesis_data/preprocessed/data-metrics.json"
  training_instances_count=$(cat thesis_data/preprocessed/data-metrics.json | jq .train)
  echo "[*] Using $training_instances_count as training instances"
  set -u

  ./scripts/run-exp-simple_thesis.sh \
  -c experiment_configs/original.jsonnet \
  -s $OUTPUT_PATH/ \
  --num-epochs 2 \
  --batch-size 4 \
  --train-path thesis_data/preprocessed/data-train.p \
  --dev-path thesis_data/preprocessed/data-val.p \
  --bert-vocab thesis_data/finnish_bert_cased/vocab.txt \
  --bert-weights TurkuNLP/bert-base-finnish-cased-v1 \
  --vocab thesis_data/finnish_bert_cased/vocabulary \
  --num-train-instances $training_instances_count \
  --cuda-device -1
else

  echo "[*] Using DEMO samples for training.. ====[NOT THE REAL FULL DATA]===="
  OUTPUT_PATH="thesis-model-output-demo"
  rm -r $OUTPUT_PATH
  echo "[*] Removed data with rm -r  ${OUTPUT_PATH}"

  echo "[*] Make sure you have a correct number or training instances"
  echo "[*] Reading number of training instances from thesis_data/preprocessed_demo/data-metrics.json"
  training_instances_count=$(cat thesis_data/preprocessed_demo/data-metrics.json | jq .train)
  echo "[*] Using $training_instances_count as training instances"
  set -u
  ./scripts/run-exp-simple_thesis.sh \
  -c experiment_configs/original.jsonnet \
  -s $OUTPUT_PATH/ \
  --num-epochs 2 \
  --batch-size 2 \
  --train-path thesis_data/preprocessed_demo/data-train.p \
  --dev-path thesis_data/preprocessed_demo/data-val.p \
  --bert-vocab thesis_data/finnish_bert_cased/vocab.txt \
  --bert-weights thesis_data/finnish_bert_cased/finnish_bert.tar.gz \
  --vocab thesis_data/finnish_bert_cased/vocabulary \
  --num-train-instances $training_instances_count \
  --cuda-device -1
fi
#--bert-weights thesis_data/finnish_bert_cased/finnish_bert.tar.gz \


echo "[*] DONE. Training finished.."
echo "[*] Next Step: See finetuned model in $OUTPUT_PATH"
