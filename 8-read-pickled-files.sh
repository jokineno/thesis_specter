#!/bin/bash

echo "Starting to unpickle pickeld training files.."

python read_pickled_files.py \
 --dir thesis_data/preprocessed_demo/ \
 --output_path theis_data/preprocessed_demo/unpickled \

echo "Done"