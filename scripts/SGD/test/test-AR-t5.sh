#!/bin/bash

EXPERIMENT_NAME="replication/AR-T5-SGstar-SGD-lr=1e-4-b=8-ep=5"
CHECKPOINT_NAME="epoch=4-step=103115.ckpt"

# Greedy
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config "configs/SGD/t5-SchemaGuided_config.jsonnet" \
    --mode "test" \
    --opts \
    executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
    test_suffix="greedy-ep=5-bsize=1" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name=$CHECKPOINT_NAME \
    test.batch_size=1 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1 \

# Beam Search
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config "configs/SGD/t5-SchemaGuided_config.jsonnet" \
    --mode "test" \
    --opts \
    executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
    executor.init_kwargs.use_constrained_decoding=True \
    test_suffix="constrained_beamsearch-bs=4-ep=5-batchsize=1" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name=$CHECKPOINT_NAME \
    test.batch_size=1 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1 \
    test.generate_params.num_beams=4 \
