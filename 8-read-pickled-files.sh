#!/bin/bash

echo "Starting to unpickle pickeld training files.."

python read_pickled_files.py \
 --dir thesis_data/preprocessed_demo/ \
 --output_path thesis_data/preprocessed_demo/unpickled \

echo "Done"