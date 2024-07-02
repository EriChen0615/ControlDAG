local meta = import 'example_meta_config.libsonnet';
local data = import 'example_data_config.libsonnet';

{
    experiment_name: 'trial-MRPC',
    test_suffix: 'v1',
    meta: meta.default_meta,
    data_pipeline: data.MRPC_datapipeline,
    executor: {
        ExecutorClass: 'BertMRPCExecutor',
        init_kwargs: {},
        model_config: {
            bert_model_version: 'bert-base-uncased'
        },
    },
    train: {
        batch_size: 8,
        trainer_paras: {
            max_epochs: 5,
            log_every_n_steps: 30,
            check_val_every_n_epoch: 1,
        },
        model_checkpoint_callback_paras: {
            monitor: 'val_loss',
            save_top_k: -1
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
        trainer_paras: {},
        batch_size: 64,
    },
    eval: {
        'eval_op_name': 'Your eval op name'
    },
}


