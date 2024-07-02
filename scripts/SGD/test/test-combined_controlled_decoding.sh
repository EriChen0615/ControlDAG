#!/bin/bash
#SBATCH -J test-dag2wfsa 
#SBATCH -A BYRNE-SL3-GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --array=0-3
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere
#!SBATCH -p cclake
#! ############################################################

# SchemaGuided SR 
EXPERIMENT_NAME="replication/SGD-DA-T5-SGstar-upsample=5-use_glat=True-lr=1e-4-b=8-ep=40"
CHECKPOINT_NAME="epoch=29-step=618690.ckpt"
EPOCH_NUM=${CHECKPOINT_NAME#*=}   # Removes everything up to and including 'epoch='
EPOCH_NUM=${EPOCH_NUM%-*}         # Removes '-step=...' and everything after

echo "EPOCH_NUM: $EPOCH_NUM"



LOG=/dev/stdout
ERR=/dev/stderr

# Uncomment below to submit with `sbatch`
# . /etc/profile.d/modules.sh                # Leave this line (enables the module command)
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

CONFIG_FILE="configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" # Schema Guided from Kale (2022)
# CONFIG_FILE="configs/SGD/da-t5-SchemaGuided_config.jsonnet" # Improved Schema Guided from Chen (2023)
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
# DAG Pruning Setting
exp0[TOP_P_TRANSITION]=3
exp0[TOP_K_EMISSION]=3
exp0[ADD_FORCED_TOKENS_TO_DAG]=True # naive, False
exp0[NUM_FIRST_HLC_TOKEN]=0.3

# Vocabulary Constraint Setting
exp0[USE_VC]=True
exp0[DYN_VOCAB]=True
exp0[VOCAB_VER]="6"
exp0[VOCAB_PERCENT]="90"

# Hard Lexical Constraint Setting
exp0[USE_HLC]=True

# Length Constraint Setting
exp0[WIP]=0.0
exp0[USE_LENGTH_CONSTRAINT]=none # regressor, none
exp0[PRUNE_CUMPROB]=0.7
exp0[LEN_CONSTRAINT_RANKING]=exact_length # exact_length, min_norm_dist_with_penalty
exp0[LEN_SEARCH_ALGO]=dfs_memo
exp0[A_VAL]=1


# Assign value to exp1
# DAG Pruning Setting
exp1[TOP_P_TRANSITION]=3
exp1[TOP_K_EMISSION]=3
exp1[ADD_FORCED_TOKENS_TO_DAG]=True # naive, False
exp1[NUM_FIRST_HLC_TOKEN]=0.3

# Vocabulary Constraint Setting
exp1[USE_VC]=True
exp1[DYN_VOCAB]=True
exp1[VOCAB_VER]="6"
exp1[VOCAB_PERCENT]="90"

# Hard Lexical Constraint Setting
exp1[USE_HLC]=True

# Length Constraint Setting
exp1[WIP]=0.0
exp1[USE_LENGTH_CONSTRAINT]=regressor # regressor, none
exp1[PRUNE_CUMPROB]=0.7
exp1[LEN_CONSTRAINT_RANKING]=min_norm_dist_with_penalty # exact_length, min_norm_dist_with_penalty
exp1[LEN_SEARCH_ALGO]=dfs_memo
exp1[A_VAL]=1

# Assign value to exp2
# DAG Pruning Setting
exp2[TOP_P_TRANSITION]=3
exp2[TOP_K_EMISSION]=3
exp2[ADD_FORCED_TOKENS_TO_DAG]=True # naive, False
exp2[NUM_FIRST_HLC_TOKEN]=0.3

# Vocabulary Constraint Setting
exp2[USE_VC]=True
exp2[DYN_VOCAB]=True
exp2[VOCAB_VER]="6"
exp2[VOCAB_PERCENT]="90"

# Hard Lexical Constraint Setting
exp2[USE_HLC]=True

# Length Constraint Setting
exp2[WIP]=-0.5
exp2[USE_LENGTH_CONSTRAINT]=none # regressor, none
exp2[PRUNE_CUMPROB]=0.7
exp2[LEN_CONSTRAINT_RANKING]=min_norm_dist_with_penalty # exact_length, min_norm_dist_with_penalty
exp2[LEN_SEARCH_ALGO]=none
exp2[A_VAL]=1

# exps=(exp0 exp1)
exps=(exp2)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB

for expKey in "${exps[@]}"; do
    # DAG pruning
    TOP_P="${expKey}[TOP_P_TRANSITION]"
    TOP_K="${expKey}[TOP_K_EMISSION]"
    FEMIT="${expKey}[ADD_FORCED_TOKENS_TO_DAG]"
    FTR="${expKey}[NUM_FIRST_HLC_TOKEN]"

    # Vocabulary Constraint
    USE_VC="${expKey}[USE_VC]"
    VOCAB_VER="${expKey}[VOCAB_VER]"
    VOCAB_PERCENT="${expKey}[VOCAB_PERCENT]"
    dyn_vocab="${expKey}[DYN_VOCAB]"
    vocab_file="data/dstc8-schema-guided-dialogue/train_vocab-${!VOCAB_PERCENT}%-v${!VOCAB_VER}.json"

    # Hard Lexical Constraint
    USE_HLC="${expKey}[USE_HLC]"

    # Length Constraint
    WIP=${expKey}[WIP]
    USE_LENGTH_CONSTRAINT="${expKey}[USE_LENGTH_CONSTRAINT]"
    PRUNE_CUMPROB="${expKey}[PRUNE_CUMPROB]"
    LEN_CONSTRAINT_RANKING="${expKey}[LEN_CONSTRAINT_RANKING]"
    LEN_SEARCH_ALGO="${expKey}[LEN_SEARCH_ALGO]"
    A_VAL="${expKey}[A_VAL]"

    echo =================================
    TEST_NAME="2308-ControlledDecoding/p=${!TOP_P}-k=${!TOP_K}-FEMIT=${!FEMIT}(FTR=${!FTR})-HLC=${!USE_HLC}-VC=${!USE_VC}(dyn=${!dyn_vocab}-vocab=${!VOCAB_PERCENT}-v${!VOCAB_VER})-WIP=${!WIP}-LC=${!LEN_SEARCH_ALGO}(USE=${!USE_LENGTH_CONSTRAINT}-RANK=${!LEN_CONSTRAINT_RANKING}-PRUNE_P=${!PRUNE_CUMPROB}-A=${!A_VAL})-ep=${EPOCH_NUM}"
    echo "|| Test name: $TEST_NAME ||"

    echo =====Settings for DAG-Pruning=====
    echo TOP_P_Transition: ${!TOP_P}
    echo TOP_K_Emission: ${!TOP_K}
    echo Forced-Emit: ${!FEMIT}
    echo Number of first hlc token: ${!FTR}
    echo 

    echo =====Settings for Vocabulary Constraint=====
    echo WIP: ${!WIP}
    echo Use vocabulary constraint: ${!USE_VC}
    echo Dynamic vocabulary: ${!dyn_vocab}
    echo VOCAB_VER: ${!VOCAB_VER}
    echo VOCAB_PERCENT: ${!VOCAB_PERCENT}
    echo Vocabulary file: ${vocab_file}
    echo

    echo =====Settings for Hard Lexical Constraint=====
    echo USE_HLC: ${!USE_HLC}
    echo
    
    echo =====Settings for Length Constraint=====
    echo USE_LENGTH_CONSTRAINT: ${!USE_LENGTH_CONSTRAINT}
    echo PRUNE_CUMPROB: ${!PRUNE_CUMPROB}
    echo LEN_CONSTRAINT_RANKING: ${!LEN_CONSTRAINT_RANKING}
    echo LEN_SEARCH_ALGO: ${!LEN_SEARCH_ALGO}
    echo A_VAL: ${!A_VAL} 
    echo
    

    # python -m pdb -c continue src/main.py \
    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config $CONFIG_FILE \
        --mode "test" \
        --opts \
        executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
        executor.init_kwargs.dump_dag=False \
        executor.init_kwargs.use_length_constraint=${!USE_LENGTH_CONSTRAINT} \
        executor.init_kwargs.length_regressor_coeff=["26.07857915","0.3855"] \
        test_suffix=$TEST_NAME \
        exp_version="0" \
        meta.logger_enable=["csv"] \
        model_config.gen_decoder_input_args.upsample_scale=5 \
        test.checkpoint_name=$CHECKPOINT_NAME \
        test.batch_size=1 \
        test.generate_params.decode_params.top_k_emissions=${!TOP_K} \
        test.generate_params.decode_params.top_k_transitions=${!TOP_P} \
        test.generate_params.decode_params.use_constraints=${!USE_HLC} \
        test.generate_params.decode_params.add_forced_tokens_to_dag=${!FEMIT} \
        test.generate_params.decode_params.num_enforced_first_sv_tokens=0.3 \
        test.generate_params.decode_params.apply_vocab_constraint=fsa \
        test.generate_params.decode_params.vocab_file=${vocab_file} \
        test.generate_params.decode_params.add_vocab_dynamically=${!dyn_vocab} \
        test.generate_params.decode_params.word_insertion_penalty=${!WIP} \
        test.generate_params.decode_params.len_search_algo=${!LEN_SEARCH_ALGO} \
        test.generate_params.decode_params.len_prune_cumprob=${!PRUNE_CUMPROB} \
        test.generate_params.decode_params.len_constraint_ranking=${!LEN_CONSTRAINT_RANKING} \
        test.generate_params.decode_params.len_strictness_A=${!A_VAL} \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 \
        > $LOG 2> $ERR
        # executor.init_kwargs.length_regressor_coeff=["20.0","0.3855"] \
        # --config "configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" \
done