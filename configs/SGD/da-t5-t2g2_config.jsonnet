local data = import '../data_config.libsonnet';
local exec = import '../executor_config.libsonnet';
local meta = import '../meta_config.libsonnet';
local eval = import '../eval_config.libsonnet';


{
  experiment_name: 'DA-T5-with-bos',
  test_suffix: '-trial',
  meta: meta.default_meta,
  data_pipeline: data.GEMSGD_exp_data_pipeline,
  executor: exec.da_executor_config,
  model_config: exec.da_transformer_model_config,
  tokenizer_config: data.t5_tokenizer_with_bos_config,
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
      save_last: true,
      every_n_epochs: 5,
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
    batch_size: 1,
    generate_params: {
      decode_strategy: 'fst_shortest_path',
      decode_params: {
          tokenizer_config: data.t5_tokenizer_with_bos_config,
          top_k_emissions: 1,
          top_k_transitions: 2,
      },
      // decode_strategy: 'nar_greedy',
      // decode_params: {
      // },
    },
  },
  eval: {
    pipeline_config: eval.SGD_eval_pipeline
    // {
    //   DataPipelineLib: 'data_modules',
    //   DataPipelineClass: 'DataPipeline',
    //   name: 'EvaluationPipeline',
    //   regenerate: true,
    //   do_inspect: true,
    //   transforms: {
    //     'input:GetInferEvalRecorder': {
    //       transform_name: 'GetEvaluationRecorder',
    //       setup_kwargs: {},
    //       regenerate: false,
    //       cache: false,
    //     },
    //     'input:GetGEMSGDTestReferences': {
    //       transform_name: 'GetGEMSGDTestReferences',
    //       regenerate: false,
    //       cache: false,
    //     },
    //     'process:ComputeBLEUScore': {
    //       input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
    //       transform_name: 'ComputeBLEU',
    //       setup_kwargs: {},
    //       regenerate: false,
    //       cache: false,
    //     },
    //     'process:ComputeSlotErrorRate': {
    //       input_node: 'input:GetInferEvalRecorder',
    //       transform_name: 'ComputeSER',
    //       setup_kwargs: {},
    //       regenerate: false,
    //       cache: false,
    //     },
    //     'process:ComputeDatasetLevelStats': {
    //       input_node: 'input:GetInferEvalRecorder',
    //       transform_name: 'ComputeAverages',
    //       setup_kwargs: {
    //         fields_to_average: ['decoded_score', 'token_recall']
    //       },
    //       regenerate: false,
    //       cache: false,
    //     },
    //     'output:MergeAndOutputFinalEvaluation': {
    //       input_node: ['process:ComputeBLEUScore', 'process:ComputeSlotErrorRate', 'process:ComputeDatasetLevelStats'],
    //       transform_name: 'MergeAllEvalRecorderAndSave',
    //       setup_kwargs: {},
    //       regenerate: false,
    //       cache: false,
    //     },
    //     'output:DisplayEvaluationResults': {
    //       input_node: 'output:MergeAndOutputFinalEvaluation',
    //       transform_name: 'DisplayEvalResults',
    //       setup_kwargs: {
    //         rows_to_display: 5,
    //         display_format: 'csv',
    //       },
    //       regenerate: false,
    //       cache: false,
    //     },
    //   },
    // },
  },
}
