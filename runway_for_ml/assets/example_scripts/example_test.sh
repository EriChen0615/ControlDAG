#!/bin/bash

python src/main.py \
    --config "configs/MRPC_config.jsonnet" \
    --mode "test"
    --opts \
    test_suffix="v1" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name="epoch=4.ckpt" \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1