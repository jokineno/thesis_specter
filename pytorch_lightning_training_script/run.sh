#!/bin/bash 

#export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mkdir -p save 
mkdir -p data 

python train.py \
--save_dir ./save \
--gpus 0 \
--train_file ./data/data-train.p \
--dev_file ./data/data-val.p \
--test_file ./data/data-test.p \
--batch_size 32 \
--num_workers 10 \
--num_epochs 1 \
--grad_accum 256