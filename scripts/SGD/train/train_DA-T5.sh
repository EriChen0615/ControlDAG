#!/bin/bash
LOG=/dev/stdout
ERR=/dev/stderr

# Uncomment below to submit with `sbatch`
# . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
# module purge                               # Removes all modules still loaded
# module load rhel8/default-amp
# module load cuda/11.1 intel/mkl/2017.4
# module load miniconda/3
# eval "$(conda shell.bash hook)"

# source scripts/activate_env.sh
# export OMP_NUM_THREADS=1

echo which python

LOG_DIR=experiments/log-$EXPERIMENT_NAME 
[ -d "$LOG_DIR" ] && echo "$LOG_DIR exists" || mkdir $LOG_DIR

UPSAMPLE_SCALE=5
EXPERIMENT_NAME="0915-SGD-DA-T5-SGstar(no_bos)-upsample=${UPSAMPLE_SCALE}-lr=1e-4-b=8-ep=10->20"
CHECKPOINT_NAME="experiments/AA2309/0915-SGD-DA-T5-SGstar(no_bos)-upsample=5-lr=1e-4-b=8-ep=0->10_V0/train/saved_models/epoch=9-step=206230.ckpt"
# Train DA-Transformer with NAR-T5 from scratch (or continue from checkpoint), on the full SGD
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config "configs/SGD/da-t5-SchemaGuided_config.jsonnet" \
    --mode "train" \
    --opts \
    meta.EXPERIMENT_FOLDER="experiments/AA2309/" \
    meta.logger_enable="[\"tensorboard\"]" \
    meta.WANDB.tags="[\"SGD\", \"DA-T5\", \"Trial\"]" \
    meta.WANDB.group="SGD-DA-T5-training" \
    model_config.gen_decoder_input_args.upsample_scale=${UPSAMPLE_SCALE} \
    executor.init_kwargs.use_data_node=output:SchemaGuidedWithForcedTokens \
    executor.model_config.use_pretrained_base=False \
    executor.model_config.use_pretrained_encoder=False \
    executor.model_config.base_model_class=NAR_T5 \
    train.batch_size=8 \
    train.trainer_paras.max_epochs=20 \
    train.trainer_paras.accelerator="gpu" \
    train.trainer_paras.devices=1 \
    train.trainer_paras.log_every_n_steps=50 \
    train.trainer_paras.check_val_every_n_epoch=2 \
    train.trainer_paras.num_sanity_val_steps=0 \
    train.dataloader_workers=32 \
    test.batch_size=1 \
    test.generate_params.decode_strategy="nar_greedy" \
    >> $LOG 2> $ERR