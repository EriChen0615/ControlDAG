local eval = import 'dart_eval_config.libsonnet';
local meta = import '../meta_config.libsonnet';
local data = import 'dart_data_config.libsonnet';
local exec = import 'dart_executor_config.libsonnet';


{
  experiment_name: 'DA-T5-with-bos',
  test_suffix: '-trial',
  meta: meta.default_meta,
  data_pipeline: data.GEMDART_data_pipeline,
  executor: exec.t5_executor_config,
  model_config: exec.t5_model_config,
  tokenizer_config: data.t5_tokenizer_with_bos_and_graph_config,
  train: {
    batch_size: 8,
    trainer_paras: {
      max_epochs: 50,
      log_every_n_steps: 30,
      check_val_every_n_epoch: 1,
    },
    model_checkpoint_callback_paras: {
      monitor: 'val_loss',
      save_top_k: -1,
      every_n_epochs: 1,
      save_last: true,
    },
    optimizer_config: {
      optimizer_name: 'AdamW',
      optimizer_params: {
        lr: 0.0001,
      },
    },
  },
  test: {
    checkpoint_name: 'epoch=4-step=1220.ckpt',
    save_decode: true,
    trainer_paras: {},
    batch_size: 64,
    generate_params: {
        max_new_tokens: 500,
    },
  },
  eval: {
    pipeline_config: eval.DART_eval_pipeline,
    // pipeline_config: eval.DART_neo_only_eval_pipeline,
  },
}
