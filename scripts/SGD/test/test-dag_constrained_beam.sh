#!/bin/bash
EXPERIMENT_NAME="replication/SGD-DA-T5-SGstar-upsample=5-use_glat=True-lr=1e-4-b=8-ep=40"
CHECKPOINT_NAME="epoch=29-step=618690.ckpt"

LOG=/dev/stdout
ERR=/dev/stderr

A_VAL=1
APPLY_VOCAB_CON=False
USE_LENGTH_CONSTRAINT=none
LEN_CONSTRAINT_RANKING=min_norm_dist_with_penalty
PRUNE_CUMPROB=0.7
TOP_P=2
TOP_K=3
# LEN_CONSTRAINT_RANKING=exact_length
# USE_LENGTH_CONSTRAINT=regressor

declare -A exp1
declare -A exp2
declare -A exp3
declare -A exp4
declare -A exp5
declare -A exp6

# Assign value to exp1
exp1[USE_DYN]=True
exp1[BEAMSIZE]=5

# Assign value to exp2
exp2[USE_DYN]=False
exp2[BEAMSIZE]=5

# Assign value to exp3

# exps=(exp1 exp2)
exps=(exp1 exp2)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB


for expKey in "${exps[@]}"; do
    BEAMSIZE="${expKey}[BEAMSIZE]"
    USE_DYN="${expKey}[USE_DYN]"
    echo BEAMSIZE: ${!BEAMSIZE}
    echo USE_DYN: ${!USE_DYN}

    TEST_NAME="2308-HLC/ConstrainedDAGBeamSearch-BEAMSIZE=${!BEAMSIZE}-DynamicBS=${!USE_DYN}-ep=29"
    echo "Test name: $TEST_NAME"
    # python -m pdb -c continue src/main.py \
    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config "configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" \
        --mode "test" \
        --opts \
        executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
        executor.init_kwargs.dump_dag=False \
        executor.init_kwargs.use_length_constraint=$USE_LENGTH_CONSTRAINT \
        executor.init_kwargs.length_regressor_coeff=["26.07857915","0.3855"] \
        test_suffix=$TEST_NAME \
        exp_version="0" \
        meta.logger_enable=["csv"] \
        model_config.gen_decoder_input_args.upsample_scale=5 \
        test.checkpoint_name=$CHECKPOINT_NAME \
        test.batch_size=1 \
        test.generate_params.decode_strategy="constrained_dag_beam_search" \
        test.generate_params.beam_size=${!BEAMSIZE} \
        test.generate_params.decode_params={\"use_dynamic_beam_size\":${!USE_DYN}} \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 \
        > $LOG 2> $ERR
        # executor.init_kwargs.length_regressor_coeff=["20.0","0.3855"] \
done
# Constrained Beam Search (DBA) DA-Transformer
# BEAMSIZE=5
# python src/main.py \
#     --experiment_name $EXPERIMENT_NAME \
#     --config "configs/da-t5-bos.jsonnet" \
#     --mode "test" \
#     --opts \
#     executor.init_kwargs.use_data_node="output:T5-T2G2WithForcedTokens" \
#     test_suffix="constrained-beamsearch-bs=$BEAMSIZE-use_dynamic_beamsize-ep=20" \
#     test.generate_params.decode_strategy="constrained_dag_beam_search" \
#     test.generate_params.beam_size=$BEAMSIZE \
#     test.generate_params.decode_params={\"use_dynamic_beam_size\":True} \
#     exp_version="0" \
#     meta.logger_enable=["csv"] \
#     test.checkpoint_name="epoch=19-step=412460.ckpt" \
#     test.batch_size=8 \
#     test.trainer_paras.accelerator="gpu" \
#     test.trainer_paras.devices=1 \