#!/bin/bash 

set -x
echo "Starting to load finnish bert and sentence bert"
mkdir -p models
python load_model_from_huggingface_and_save_to_local.py --model TurkuNLP/sbert-cased-finnish-paraphrase --output models
python load_model_from_huggingface_and_save_to_local.py --model TurkuNLP/bert-base-finnish-cased-v1 --output models
echo "Done".
