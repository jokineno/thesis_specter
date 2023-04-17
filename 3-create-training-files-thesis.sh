#!/bin/bash
set +x

echo "Remember to build specter package if you have made changes to files under specter/ directory"
echo "Creating training files (thesis version).. "

export USE_HF_SPECIAL_TOKENS=True

# RE-INSTALL SPECTER PACKAGES UNDER PATH specter/
echo $*
if [[ $* == *--install_packages* ]]
then
 echo "re building specter packages..."
 set -x
 python setup.py install > /dev/null #not outputting the install lines
else
  echo "not rebuilding specter packages"
fi



# RUN THE TRAINING SCRIPT
set +x
if [[ ! $* == *--demo* ]]
then
  echo "Using full data for training..."
  python specter/data_utils/create_training_files.py \
    --data-dir thesis_data/training \
    --metadata thesis_data/training/metadata.json \
    --outdir thesis_data/preprocessed/
    --bert_vocab thesis_data/finnish_bert_cased/vocab.txt
  echo "Done. See results is thesis_data/preprocessed/"

else

  echo "Using 100 DEMO samples for training..."
  python specter/data_utils/create_training_files.py \
    --data-dir thesis_data/training_demo/ \
    --metadata thesis_data/training_demo/metadata_full.json \
    --outdir thesis_data/preprocessed_demo \
    --bert_vocab thesis_data/finnish_bert_cased/vocab.txt
  echo "Done. See results is thesis_data/preprocessed_demo/"
fi

echo "Done. Exiting.."
