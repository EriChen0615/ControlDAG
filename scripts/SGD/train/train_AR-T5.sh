#!/bin/bash
EXPERIMENT_NAME="SGD-AR-T5-baseline"

LOG_DIR=experiments/log-$EXPERIMENT_NAME 
[ -d "$LOG_DIR" ] && echo "$LOG_DIR exists" || mkdir $LOG_DIR

python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config "configs/SGD/t5-SchemaGuided_config.jsonnet" \
    --mode "train" \
    --opts \
    meta.EXPERIMENT_FOLDER="experiments" \
    meta.logger_enable="[\"tensorboard\"]" \
    meta.WANDB.group="SGD-DA-T5-training" \
    meta.WANDB.tags="[\"SGD\", \"T5\"]" \
    executor.init_kwargs.use_data_node="output:SchemaGuidedTokenize" \
    executor.init_kwargs.dump_dag=False \
    train.trainer_paras.max_epochs=5 \
    train.trainer_paras.accelerator="gpu" \
    train.trainer_paras.num_sanity_val_steps=10 \
    train.trainer_paras.devices=1 \
    # > $LOG 2> $ERR
