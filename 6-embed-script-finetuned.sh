#!/bin/bash


# Read sample data like this. .["field"] is required with jq
# cat data/sample-metadata.json | jq '.["008b5715a2e3a52674edc325853577de86588681"]'
#

# Read output data like this
# cat output.jsonl | jq 'select(.paper_id=="<paper_id>")'

set -x
echo "Embedding samples with baseline model..."
python scripts/embed.py \
--ids data/sample.ids --metadata data/sample-metadata.json \
--model ./model-output/model.tar.gz \
--output-file output_finetuned.jsonl \
--vocab-dir data/vocab/ \
--batch-size 16 \
--cuda-device -1 # 0 = use GPU, -1 = use CPU

# The script is converted to shell script
# python specter/predict_command.py predict
# ./model.tar.gz
# data/sample.ids
# --include-package specter
# --predictor specter_predictor
# --overrides "{'model':{'predict_mode':'true','include_venue':'false'},'dataset_reader':{'type':'specter_data_reader','predict_mode':'true','paper_features_path':'data/sample-metadata.json','included_text_fields': 'abstract title'},'vocabulary':{'directory_path':'data/vocab/'}}" --cuda-device -1 --output-file output.jsonl --batch-size 16 --silent

echo "Done.."