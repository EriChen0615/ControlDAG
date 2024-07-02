#!/bin/bash
EXPERIMENT_NAME="replication/SGD-DA-T5-SGstar-upsample=5-use_glat=True-lr=1e-4-b=8-ep=40" # SG* With GLAT
CHECKPOINT_NAME="epoch=39-step=824920.ckpt" # epoch 40

LOG=/dev/stdout
ERR=/dev/stderr

CONFIG_FILE=configs/SGD/da-t5-SchemaGuided_config.jsonnet # For Improved Schema Guided Semantic Represetation in Chen (2023) Paper
TEST_FOLDER="2308-BASELINES"

# greedy decode
TEST_NAME="${TEST_FOLDER}/greedy-ep=39"
echo TEST_NAME: $TEST_NAME
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config ${CONFIG_FILE} \
    --mode "test" \
    --opts \
    test_suffix=$TEST_NAME \
    test.generate_params.decode_strategy="nar_greedy" \
    executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
    exp_version="0" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name=$CHECKPOINT_NAME \
    test.batch_size=1 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1 \

# BEAM SEARCH
declare -A beam_exp1
declare -A beam_exp2
declare -A beam_exp3

beam_exp1[BEAMSIZE]=5
beam_exp2[BEAMSIZE]=10
beam_exp3[BEAMSIZE]=20

# beam_exps=(beam_exp1 beam_exp2 beam_exp3)
beam_exps=(beam_exp1)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB
for expKey in "${beam_exps[@]}"; do
    BEAMSIZE="${expKey}[BEAMSIZE]"
    echo BEAMSIZE: ${!BEAMSIZE}

    TEST_NAME="${TEST_FOLDER}/DAGBeamSearch-BEAMSIZE=${!BEAMSIZE}-ep=39"
    echo "Test name: $TEST_NAME"
    # python -m pdb -c continue src/main.py \
    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config $CONFIG_FILE \
        --mode "test" \
        --opts \
        executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
        executor.init_kwargs.dump_dag=False \
        test_suffix=$TEST_NAME \
        exp_version="0" \
        meta.logger_enable=["csv"] \
        model_config.gen_decoder_input_args.upsample_scale=5 \
        test.checkpoint_name=$CHECKPOINT_NAME \
        test.batch_size=1 \
        test.generate_params.decode_strategy="dag_beam_search" \
        test.generate_params.beam_size=${!BEAMSIZE} \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 \
        > $LOG 2> $ERR
done


# WFSA Shortest path

declare -A wfsa_exp0
declare -A wfsa_exp1

wfsa_exp0[TOP_P_TRANSITION]=3
wfsa_exp0[TOP_K_EMISSION]=3

wfsa_exp1[TOP_P_TRANSITION]=5
wfsa_exp1[TOP_K_EMISSION]=3

# wfsa_exps=(wfsa_exp0 wfsa_exp1)
wfsa_exps=(wfsa_exp0 wfsa_exp1)

# expKey=${exps[$SLURM_ARRAY_TASK_ID]} # for SLURM ARRAY JOB

for expKey in "${wfsa_exps[@]}"; do
    TOP_P="${expKey}[TOP_P_TRANSITION]"
    TOP_K="${expKey}[TOP_K_EMISSION]"
    echo TOP_P_Transition: ${!TOP_P}
    echo TOP_K_Emission: ${!TOP_K}

    TEST_NAME="${TEST_FOLDER}/p=${!TOP_P}-k=${!TOP_K}-WFSA_shortest-HLC=False-VC=False-LC=None-ep=39"
    echo "Test name: $TEST_NAME"
    # python -m pdb -c continue src/main.py \
    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config "configs/SGD/da-t5-paperSchemaGuided_config.jsonnet" \
        --mode "test" \
        --opts \
        executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
        executor.init_kwargs.dump_dag=False \
        test_suffix=$TEST_NAME \
        exp_version="0" \
        meta.logger_enable=["csv"] \
        test.checkpoint_name=$CHECKPOINT_NAME \
        test.batch_size=1 \
        test.generate_params.decode_params.use_constraints=False \
        test.generate_params.decode_params.apply_vocab_constraint=False \
        test.generate_params.decode_params.add_forced_tokens_to_dag=False \
        test.generate_params.decode_params.len_search_algo="none" \
        test.generate_params.decode_params.top_k_emissions=${!TOP_K} \
        test.generate_params.decode_params.top_k_transitions=${!TOP_P} \
        model_config.gen_decoder_input_args.upsample_scale=5 \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 \
        > $LOG 2> $ERR
done
