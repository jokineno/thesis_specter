#!/bin/bash


set -x

echo "Starting to embed"
python scripts/embed_papers_hf.py \
  --data-path thesis_data/inference_demo/citations/sample-metadata.json \
  --output ./thesis_data/inference_demo/citations/sbert/user-citations.jsonl \
  --model_and_tokenizer TurkuNLP/sbert-cased-finnish-paraphrase
echo "Done"