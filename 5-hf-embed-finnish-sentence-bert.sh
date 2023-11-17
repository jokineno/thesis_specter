#!/bin/bash


set -x
set -u
echo "[* ]Starting to embed files in thesis_data/inference_demo/paper-metadata-cls.json"
MODEL="TurkuNLP/sbert-cased-finnish-paraphrase"
echo "Using HuggingFace model $MODEL"

ts=$(date +%Y%m%d_%H%M%S)
output_filename="${ts}_baseline_sbert_cls.jsonl"

python scripts/embed_papers_hf.py \
  --data-path "./thesis_data/inference_demo/paper-metadata-cls.json" \
  --output "./thesis_data/inference_demo/results/$output_filename" \
  --model_and_tokenizer $MODEL
echo "Done"