#!/bin/bash
set -x

echo "Creating training files (thesis version).. "

if [[ ! $* == *--demo* ]]
then

  echo "Using full data for training..."
  python specter/data_utils/create_training_files.py \
    --data-dir thesis_data/training \
    --metadata thesis_data/training/metadata.json \
    --outdir thesis_data/preprocessed/
  echo "Done. See results is thesis_data/preprocessed/"

else

  echo "Using demo samples for training..."
  python specter/data_utils/create_training_files.py \
    --data-dir thesis_data/training_demo/ \
    --metadata thesis_data/training_demo/metadata.json \
    --outdir thesis_data/preprocessed_demo \
    --bert_vocab thesis_data/finnish_bert_cased/vocab.txt
  echo "Done. See results is thesis_data/preprocessed_demo/"
fi

echo "Done. Exiting.."