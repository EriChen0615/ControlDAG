#!/bin/bash
EXPERIMENT_NAME="replication/SGD-DA-T5-SGstar-upsample=5-use_glat=True-lr=1e-4-b=8-ep=40" # SG* With GLAT
CHECKPOINT_NAME="epoch=39-step=824920.ckpt" # epoch 40
# CHECKPOINT_NAME="last.ckpt" # epoch 50
MODE=test

LOG=/dev/stdout
ERR=/dev/stderr
TEST_FOLDER="BASELINES-ep=40"

# lookahead decode
TEST_NAME="${TEST_FOLDER}/lookahead"
echo TEST_NAME: $TEST_NAME
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config ${CONFIG_FILE} \
    --mode $MODE \
    --opts \
    test_suffix=$TEST_NAME \
    test.generate_params.decode_strategy="lookahead" \
    executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name=$CHECKPOINT_NAME \
    test.batch_size=1 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1 \

# viterbi decode
TEST_NAME="${TEST_FOLDER}/viterbi"
echo TEST_NAME: $TEST_NAME
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config ${CONFIG_FILE} \
    --mode $MODE \
    --opts \
    test_suffix=$TEST_NAME \
    test.generate_params.decode_strategy="viterbi" \
    executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name=$CHECKPOINT_NAME \
    test.batch_size=1 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1 \

TEST_NAME="${TEST_FOLDER}/joint_viterbi"
echo TEST_NAME: $TEST_NAME
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config ${CONFIG_FILE} \
    --mode $MODE \
    --opts \
    test_suffix=$TEST_NAME \
    test.generate_params.decode_strategy="joint_viterbi" \
    executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name=$CHECKPOINT_NAME \
    test.batch_size=1 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1 \