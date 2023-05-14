#!/bin/bash


# Read sample data like this. .["field"] is required with jq
# cat data/sample-metadata.json | jq '.["008b5715a2e3a52674edc325853577de86588681"]'
#

# Read output data like this
# cat output.jsonl | jq 'select(.paper_id=="<paper_id>")'

set -x
export USE_HF_SPECIAL_TOKENS=True
echo "Embedding samples with finetuned model..."

python scripts/embed.py \
--ids thesis_data/inference_demo/citations/sample.ids \
--metadata thesis_data/inference_demo/citations/sample-metadata.json \
--model ./thesis-model-output/model.tar.gz \
--output-file ./thesis_data/inference_demo/citations/myownfinetuned/user-citation.jsonl \
--vocab-dir thesis_data/vocab/finnish_bert_cased/vocabulary/ \
--batch-size 16 \
--cuda-device -1 # 0 = use GPU, -1 = use CPU

echo "Done.."
