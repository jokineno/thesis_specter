#!/bin/bash


# Read sample data like this. .["field"] is required with jq
# cat data/sample-metadata.json | jq '.["008b5715a2e3a52674edc325853577de86588681"]'
#

# Read output data like this
# cat output.jsonl | jq 'select(.paper_id=="<paper_id>")'

set -x
export USE_HF_SPECIAL_TOKENS=True


ts=$(date +%Y%m%d_%H%M%S)
output_filename="${ts}_baseline_finnishbert_cls_from_allennlp.jsonl"

if [[ ! $* == *--demo* ]]
then
  echo "[*] FULL Embedding samples with baseline model..."
  OUTPUT_FILE="./thesis_data/inference/results/$output_filename"
  python scripts/embed.py \
  --ids thesis_data/inference/sample.ids \
  --metadata thesis_data/inference/sample-metadata.json \
  --model ./thesis_data/inference/models/finnish_bert_base_cased_v1/finnish_bert.tar.gz \
  --output-file $OUTPUT_FILE \
  --vocab-dir thesis_data/finnish_bert_cased/vocabulary/ \
  --batch-size 16 \
  --cuda-device -1
  # 0 = use GPU, -1 = use CPU
else
  echo "[*] DEMO Embedding samples with baseline model..."
  OUTPUT_FILE="./thesis_data/inference_demo/results/$output_filename"
  python scripts/embed.py \
  --ids thesis_data/inference_demo/sample.ids \
  --metadata thesis_data/inference_demo/paper-metadata-cls.json \
  --model ./thesis_data/inference/models/finnish_bert_base_cased_v1/finnish_bert.tar.gz \
  --output-file $OUTPUT_FILE \
  --vocab-dir thesis_data/finnish_bert_cased/vocabulary/ \
  --batch-size 16 \
  --cuda-device -1
    # 0 = use GPU, -1 = use CPU
fi

echo "[*] Done embedding samples with baseline model..."
echo "[*] Output file: $OUTPUT_FILE"

# The script is converted to shell script
# python specter/predict_command.py predict
# ./model.tar.gz
# data/sample.ids
# --include-package specter
# --predictor specter_predictor
# --overrides "{'model':{'predict_mode':'true','include_venue':'false'},'dataset_reader':{'type':'specter_data_reader','predict_mode':'true','paper_features_path':'data/sample-metadata.json','included_text_fields': 'abstract title'},'vocabulary':{'directory_path':'data/vocab/'}}" --cuda-device -1 --output-file output.jsonl --batch-size 16 --silent

echo "Done.."
