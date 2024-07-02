#!/bin/bash
EXPERIMENT_NAME="DA-T5-SchemaGuided-SGD_l=5-lr=1e-4-b=8-e=30"
CHECKPOINT_NAME="epoch=29-step=618690.ckpt"

LOG=/dev/stdout
ERR=/dev/stderr

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



exp0[USE_LENGTH_CONSTRAINT]=oracle
exp0[PRUNE_CUMPROB]=0.7
exp0[LEN_CONSTRAINT_RANKING]=exact_length
exp0[LEN_SEARCH_ALGO]=dfs_memo
exp0[A_VAL]=1

# Assign value to exp1
exp1[USE_LENGTH_CONSTRAINT]=oracle
exp1[PRUNE_CUMPROB]=0.7
exp1[LEN_CONSTRAINT_RANKING]=min_norm_dist_with_penalty
exp1[LEN_SEARCH_ALGO]=dfs_memo
exp1[A_VAL]=1

exp2[USE_LENGTH_CONSTRAINT]=oracle
exp2[PRUNE_CUMPROB]=0.7
exp2[LEN_CONSTRAINT_RANKING]=min_norm_dist_with_penalty
exp2[LEN_SEARCH_ALGO]=dfs_memo
exp2[A_VAL]=1

exp3[USE_LENGTH_CONSTRAINT]=regressor
exp3[PRUNE_CUMPROB]=0.7
exp3[LEN_CONSTRAINT_RANKING]=min_norm_dist_with_penalty
exp3[LEN_SEARCH_ALGO]=dfs_memo
exp3[A_VAL]=1

# exps=(exp1 exp2 exp3 exp4)
exps=(exp3)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB

for expKey in "${exps[@]}"; do
    USE_LENGTH_CONSTRAINT="${expKey}[USE_LENGTH_CONSTRAINT]"
    PRUNE_CUMPROB="${expKey}[PRUNE_CUMPROB]"
    LEN_CONSTRAINT_RANKING="${expKey}[LEN_CONSTRAINT_RANKING]"
    LEN_SEARCH_ALGO="${expKey}[LEN_SEARCH_ALGO]"
    A_VAL="${expKey}[A_VAL]"

    echo USE_LENGTH_CONSTRAINT: ${!USE_LENGTH_CONSTRAINT}
    echo PRUNE_CUMPROB: ${!PRUNE_CUMPROB}
    echo LEN_CONSTRAINT_RANKING: ${!LEN_CONSTRAINT_RANKING}
    echo LEN_SEARCH_ALGO: ${!LEN_SEARCH_ALGO}
    echo A_VAL: ${!A_VAL}

    TEST_NAME="2308-LC/p=${TOP_P}-k=${TOP_K}-HLC=False-VC=False-LC=${!LEN_SEARCH_ALGO}(USE=${!USE_LENGTH_CONSTRAINT}-RANK=${!LEN_CONSTRAINT_RANKING}-PRUNE_P=${!PRUNE_CUMPROB}-A=${!A_VAL})-ep=29"
    echo "Test name: $TEST_NAME"
    # python -m pdb -c continue src/main.py \
    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config "configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" \
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
        test.generate_params.decode_params.use_constraints=False \
        test.generate_params.decode_params.add_forced_tokens_to_dag=False \
        test.generate_params.decode_params.top_k_emissions=${TOP_K} \
        test.generate_params.decode_params.top_k_transitions=${TOP_P} \
        test.generate_params.decode_params.apply_vocab_constraint=False \
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