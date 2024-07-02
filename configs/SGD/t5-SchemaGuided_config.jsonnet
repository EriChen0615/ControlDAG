local meta = import '../meta_config.libsonnet';
local data = import '../data_config.libsonnet';
local exec = import '../executor_config.libsonnet';
local eval = import '../eval_config.libsonnet';


{
    experiment_name: "debug-train-T5",
    test_suffix: "-trial",
    meta: meta.default_meta,
    data_pipeline: data.GEMSGD_exp_data_pipeline,
    executor: exec.t5_executor_config,
    model_config: exec.t5_model_config,
    tokenizer_config: data.t5_tokenizer_config,
    train: {
        batch_size: 8,
        trainer_paras: {
            max_epochs: 20,
            log_every_n_steps: 50,
            check_val_every_n_epoch: 1,
        },
        model_checkpoint_callback_paras: {
            monitor: 'val_loss',
            save_top_k: -1,
            every_n_epochs: 1,
            save_last: true,
        },
        optimizer_config: {
            optimizer_name: "AdamW",
            optimizer_params: {
                lr: 0.0001
            },
        },
    },
    test: {
        checkpoint_name: "epoch=4-step=1220.ckpt",
        save_decode: true,
        trainer_paras: {},
        batch_size: 64,
        generate_params: {
            max_new_tokens: 500,
            // decode_strategy: 'fst_shortest_path',
            // decode_params: {
            //     tokenizer_config: data.t5_tokenizer_with_bos_config,
            //     top_k_emissions: 3,
            //     top_k_transitions: 2,
            // },
            // decode_strategy: 'nar_greedy',
            // decode_params: {
            // },
        },
    },
    eval: {
        pipeline_config: eval.SGD_ar_eval_pipeline
  },
}