#!/bin/bash

python src/main.py \
    --config "configs/MRPC_config.jsonnet" \
    --mode 'train' \
    train.trainer_paras.accelerator="gpu" \
    train.trainer_paras.devices=1
    