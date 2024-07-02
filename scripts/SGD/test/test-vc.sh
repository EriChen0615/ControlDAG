#!/bin/bash
EXPERIMENT_NAME="DA-T5-SchemaGuided-SGD_l=5-lr=1e-4-b=8-e=30"
CHECKPOINT_NAME="epoch=29-step=618690.ckpt"



LOG=/dev/stdout
ERR=/dev/stderr

source scripts/activate_env.sh
export OMP_NUM_THREADS=1

A_VAL=1
APPLY_VOCAB_CON=False
LEN_CONSTRAINT_RANKING=min_norm_dist_with_penalty
PRUNE_CUMPROB=0.7
TOP_P=2
TOP_K=3
# LEN_CONSTRAINT_RANKING=exact_length
# USE_LENGTH_CONSTRAINT=regressor

declare -A exp0
declare -A exp1
declare -A exp2
declare -A exp3
declare -A exp4
declare -A exp5
declare -A exp6

# Assign value to exp0
exp0[TOP_P_TRANSITION]=3
exp0[TOP_K_EMISSION]=3
exp0[DYN_VOCAB]=True
exp0[ADD_FORCED_TOKENS_TO_DAG]=True
exp0[VOCAB_VER]="6"
exp0[VOCAB_PERCENT]="90"

# Assign value to exp1
exp1[TOP_P_TRANSITION]=4
exp1[TOP_K_EMISSION]=3
exp1[DYN_VOCAB]=True
exp1[ADD_FORCED_TOKENS_TO_DAG]=True
exp1[VOCAB_VER]="6"
exp1[VOCAB_PERCENT]="90"

# Assign value to exp2
exp2[TOP_P_TRANSITION]=5
exp2[TOP_K_EMISSION]=3
exp2[DYN_VOCAB]=True
exp2[ADD_FORCED_TOKENS_TO_DAG]=True
exp2[VOCAB_VER]="6"
exp2[VOCAB_PERCENT]="90"

exps=(exp0)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB

for expKey in "${exps[@]}"; do
    dyn_vocab="${expKey}[DYN_VOCAB]"
    TOP_P="${expKey}[TOP_P_TRANSITION]"
    TOP_K="${expKey}[TOP_K_EMISSION]"
    FEMIT="${expKey}[ADD_FORCED_TOKENS_TO_DAG]"
    VOCAB_VER="${expKey}[VOCAB_VER]"
    VOCAB_PERCENT="${expKey}[VOCAB_PERCENT]"
    vocab_file="/home/jc2124/rds/hpc-work/DAG-NLG-runway/data/dstc8-schema-guided-dialogue/train_vocab-${!VOCAB_PERCENT}%-v${!VOCAB_VER}.json"

    echo TOP_P_Transition: ${!TOP_P}
    echo TOP_K_Emission: ${!TOP_K}
    echo Dynamic vocabulary: ${!dyn_vocab}
    echo VOCAB_VER: ${!VOCAB_VER}
    echo VOCAB_PERCENT: ${!VOCAB_PERCENT}
    echo Vocabulary file: ${vocab_file}
    echo ForcedEmit: ${!FEMIT}
    

    TEST_NAME="2308-VC/p=${!TOP_P}-k=${!TOP_K}-HLC=False-VC=True(dyn=${!dyn_vocab}-vocab=${!VOCAB_PERCENT}-v${!VOCAB_VER}-FEMIT=${!FEMIT})-LC=None-ep=29"
    echo "Test name: $TEST_NAME"
    # python -m pdb -c continue src/main.py \
    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config "configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" \
        --mode "eval" \
        --opts \
        executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
        executor.init_kwargs.dump_dag=False \
        executor.init_kwargs.use_length_constraint=none \
        executor.init_kwargs.length_regressor_coeff=["26.07857915","0.3855"] \
        test_suffix=$TEST_NAME \
        exp_version="0" \
        meta.logger_enable=["csv"] \
        model_config.gen_decoder_input_args.upsample_scale=5 \
        test.checkpoint_name=$CHECKPOINT_NAME \
        test.batch_size=1 \
        test.generate_params.decode_params.top_k_emissions=${!TOP_K} \
        test.generate_params.decode_params.top_k_transitions=${!TOP_P} \
        test.generate_params.decode_params.use_constraints=False \
        test.generate_params.decode_params.add_forced_tokens_to_dag=${!FEMIT} \
        test.generate_params.decode_params.num_enforced_first_sv_tokens=0.3 \
        test.generate_params.decode_params.apply_vocab_constraint=fsa \
        test.generate_params.decode_params.vocab_file=${vocab_file} \
        test.generate_params.decode_params.add_vocab_dynamically=${!dyn_vocab} \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 \
        > $LOG 2> $ERR
        # executor.init_kwargs.length_regressor_coeff=["20.0","0.3855"] \
        # --config "configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" \
done