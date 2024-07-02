#!/bin/bash
EXPERIMENT_NAME="DA-T5-SchemaGuided-SGD_l=5-lr=1e-4-b=8-e=30"
CHECKPOINT_NAME="epoch=29-step=618690.ckpt"

EPOCH_NUM=${CHECKPOINT_NAME#*=}   # Removes everything up to and including 'epoch='
EPOCH_NUM=${EPOCH_NUM%-*}         # Removes '-step=...' and everything after

echo "EPOCH_NUM: $EPOCH_NUM"

LOG=/dev/stdout
ERR=/dev/stderr

# FST shortest path decoding + constraints
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
exp0[word_insertion_penalty]=-0.5
exp0[add_wip_to_start_of_word]=False

# Assign value to exp1
exp1[TOP_P_TRANSITION]=3
exp1[TOP_K_EMISSION]=3
exp1[word_insertion_penalty]=-0.25
exp1[add_wip_to_start_of_word]=False

exp2[TOP_P_TRANSITION]=3
exp2[TOP_K_EMISSION]=3
exp2[word_insertion_penalty]=-0.75
exp2[add_wip_to_start_of_word]=False

# Assign value to exp0
exp3[TOP_P_TRANSITION]=3
exp3[TOP_K_EMISSION]=3
exp3[word_insertion_penalty]=-0.5
exp3[add_wip_to_start_of_word]=True

# Assign value to exp1
exp4[TOP_P_TRANSITION]=3
exp4[TOP_K_EMISSION]=3
exp4[word_insertion_penalty]=-0.25
exp4[add_wip_to_start_of_word]=True

exp5[TOP_P_TRANSITION]=3
exp5[TOP_K_EMISSION]=3
exp5[word_insertion_penalty]=-0.75
exp5[add_wip_to_start_of_word]=True

TEST_FOLDER="2308-WIP"
CONFIG_FILE="configs/SGD/da-t5-paperSchemaGuided_config.jsonnet"
echo CONFIG_FILE=$CONFIG_FILE

# exps=(exp0 exp1 exp2 exp3 exp4 exp5)
exps=(exp2)

for expKey in "${exps[@]}"; do
    WIP_VALUE=${expKey}[word_insertion_penalty]
    ADD_WIP_TO_START_OF_WORD=${expKey}[add_wip_to_start_of_word]
    TOP_P=${expKey}[TOP_P_TRANSITION]
    TOP_K=${expKey}[TOP_K_EMISSION]

    echo TOP_P: ${!TOP_P}
    echo TOP_K: ${!TOP_K}
    echo WIP_VALUE: ${!WIP_VALUE}
    echo ADD_WIP_TO_START_OF_WORD: ${!ADD_WIP_TO_START_OF_WORD}
    TEST_NAME=${TEST_FOLDER}/p=${!TOP_P}-k=${!TOP_K}-HLC=False-VC=False-LC=None-WIP=${!WIP_VALUE}-AddWIPtoSOW=${!ADD_WIP_TO_START_OF_WORD}_ep=${EPOCH_NUM}
    echo TEST_NAME:$TEST_NAME

    python src/main.py \
        --experiment_name $EXPERIMENT_NAME \
        --config $CONFIG_FILE \
        --mode "test" \
        --opts \
        executor.init_kwargs.use_data_node="output:SchemaGuidedWithForcedTokens" \
        model_config.gen_decoder_input_args.upsample_scale=5 \
        executor.init_kwargs.dump_dag=False \
        test_suffix=$TEST_NAME \
        exp_version="0" \
        meta.logger_enable=["csv"] \
        test.checkpoint_name=$CHECKPOINT_NAME \
        test.batch_size=1 \
        test.generate_params.decode_params.use_constraints=False \
        test.generate_params.decode_params.top_k_emissions=${!TOP_K} \
        test.generate_params.decode_params.top_k_transitions=${!TOP_P} \
        test.generate_params.decode_params.word_insertion_penalty=${!WIP_VALUE} \
        test.generate_params.decode_params.add_wip_to_start_of_word=${!ADD_WIP_TO_START_OF_WORD} \
        test.generate_params.decode_params.apply_vocab_constraint=False \
        test.generate_params.decode_params.add_forced_tokens_to_dag=False \
        test.generate_params.decode_params.add_vocab_dynamically=False \
        test.trainer_paras.accelerator="gpu" \
        test.trainer_paras.devices=1 
        # test.generate_params.decode_params.vocab_file="/home/jc2124/rds/hpc-work/DAG-NLG-runway/data/dstc8-schema-guided-dialogue/all_vocab-98%+test_sv.json" \
        # test.generate_params.decode_params.apply_vocab_constraint=False \
        # test.generate_params.decode_params.add_vocab_dynamically=False \
        # test.generate_params.decode_params.vocab_file="/home/jc2124/rds/hpc-work/DAG-NLG-runway/data/dstc8-schema-guided-dialogue/train_vocab-98%.json" \
done

    