#!/bin/bash
EXPERIMENT_NAME="replication/SGD-DA-T5-SGstar-upsample=5-use_glat=True-lr=1e-4-b=8-ep=40"
CHECKPOINT_NAME="epoch=29-step=618690.ckpt"

LOG=/dev/stdout
ERR=/dev/stderr

# Uncomment below to submit with `sbatch`
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
# module purge                               # Removes all modules still loaded
# module load rhel8/default-amp
# module load cuda/11.1 intel/mkl/2017.4
# module load miniconda/3
eval "$(conda shell.bash hook)"

source scripts/activate_env.sh
export OMP_NUM_THREADS=1

JOBID=$SLURM_JOB_ID
LOG_DIR=experiments/log-$EXPERIMENT_NAME 
[ -d "$LOG_DIR" ] && echo "$LOG_DIR exists" || mkdir $LOG_DIR
# LOG=$LOG_DIR/test-log.$JOBID
# ERR=$LOG_DIR/test-err.$JOBID

A_VAL=1
APPLY_VOCAB_CON=False
USE_LENGTH_CONSTRAINT=none
LEN_CONSTRAINT_RANKING=min_norm_dist_with_penalty
PRUNE_CUMPROB=0.7
TOP_P=3
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
exp1[FORCE_EMIT]=True # naive, False
exp1[NUM_FIRST_HLC_TOKEN]=100

# Assign value to exp2
exp2[FORCE_EMIT]=True
exp2[NUM_FIRST_HLC_TOKEN]=0.3

# Assign value to exp3
exp3[FORCE_EMIT]=True
exp3[NUM_FIRST_HLC_TOKEN]=0.3

# Assign value to exp3
exp4[FORCE_EMIT]=True
exp4[NUM_FIRST_HLC_TOKEN]=0.5

# Assign value to exp3
exp5[FORCE_EMIT]=True
exp5[NUM_FIRST_HLC_TOKEN]=0.7

# exps=(exp1 exp2 exp3 exp4)
exps=(exp2)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB

for expKey in "${exps[@]}"; do
    FEMIT="${expKey}[FORCE_EMIT]"
    FTR="${expKey}[NUM_FIRST_HLC_TOKEN]"
    echo Forced-Emit: ${!FEMIT}
    echo Number of first hlc token: ${!FTR}

    TEST_NAME="2308-HLC/p=${TOP_P}-k=${TOP_K}-HLC=${!FEMIT}(FTR=${!FTR})-VC=False-LC=None-ep=29"
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
        test.generate_params.decode_params.use_constraints=True \
        test.generate_params.decode_params.add_forced_tokens_to_dag=${!FEMIT} \
        test.generate_params.decode_params.num_enforced_first_sv_tokens=${!FTR} \
        test.generate_params.decode_params.top_k_emissions=${TOP_K} \
        test.generate_params.decode_params.top_k_transitions=${TOP_P} \
        test.generate_params.decode_params.apply_vocab_constraint=$APPLY_VOCAB_CON \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 \
        > $LOG 2> $ERR
        # executor.init_kwargs.length_regressor_coeff=["20.0","0.3855"] \
done