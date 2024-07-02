// This is the configuration for models. It defines 
// 1. The model class
// 2. The model parameters
// 3. The default data input/output used
// 4. The default training/testing/validation hyperparameters
// These can be overriden in experiment configuration file, or by using commandline options

local data = import 'dart_data_config.libsonnet';

// training/testing/validation hyperparameters
local train_batch_size = 32;
local valid_batch_size = 32;
local test_batch_size = 32;
local valid_step_size = 100;
local save_interval = 1;
local break_interval = 3000;
local train_epochs = 9999;
local adam_epsilon = 1e-08;
local lr = 1e-4;
local retriever_lr = 1e-5;
local gradient_accumulation_steps = 4;
local gradient_clipping = 0;
local warmup_steps = 0;

// local seed=2021; // override as appropriate
local t5_version = 't5-small';

local T5ModelConfig = {
  model_version: t5_version,
  ModelLib: "transformers",
  ModelClass: "T5ForConditionalGeneration",
  ConfigClass: "T5Config",
  init_kwargs: {},
  load_checkpoint_kwargs: {},
  additional_kwargs: {},
};

local optimizer_config = {
  OptimizerLib: "transformers",
  OptimizerClass: "AdamW",
  optimizer_kwargs: {
    lr: 0.0001,
    gradient_accumulation_steps: 4,
    gradient_clipping: 0,
    warmup_steps: 0,
  },
};

local Adam_config = {
  OptimizerLib: "transformers",
  OptimizerClass: "Adam",
  optimizer_kwargs: {
    lr: 0.06325,
  },
};

local training_config = {
  save_interval: 1,
  valid_step_size: 100,
  max_epochs: 5,
};

local test_config = {
  checkpoint_name: "epoch=2-step=93.ckpt",
  save_decode: true,
  // decode_path: "logs/t5-small-t2g2/test-t5-small-t2g2-s10",
  generate_params: {
    num_beams: 4,
    length_penalty: 0.6,
    max_length: 500,
  },
};

local T5ExecutorConfig = {
  ExecutorClass: 'T5Executor',
  init_kwargs: {
    'use_data_node': 'output:easy_SGD_Weather_1'
  },
  model_config: T5ModelConfig,
  optimizer_config: optimizer_config, 
  training_config: training_config,
  test_config: test_config,
  additional_kwargs: {},
};

local DATransformerConfig = {
  base_model_class: 'NAR_T5',
  base_model_version: 't5-small',
  link_features: 'decoder_out:position',
  gen_decoder_input_args: {'type':'upsample', 'upsample_scale':8},
  use_glat: true,
  use_pretrained_base: false,
  use_pretrained_encoder: false,
};

local NVSBertConfig = {
  bert_model_version: 'bert-base-uncased',
  top_k: 200,
  lambda_p: 100,
};

local DATransformerExecutorConfig = {
  ExecutorClass: 'DATransformerOnDARTExecutor',
  init_kwargs: {
    'use_data_node': 'output:TokenizeLinearizedTriples'
  },
  // model_config: DATransformerConfig,
  optimizer_config: optimizer_config, 
  training_config: training_config,
  test_config: test_config,
  // tokenizer_config: data.t5_tokenizer_with_bos_config,
  additional_kwargs: {},
};

local NVSBertExecutorConfig = {
  ExecutorClass: 'NVSBertExecutor',
  init_kwargs: {
    'use_data_node': 'output:SchemaGuided_MakeVocabularySelectionTarget'
  },
  model_config: NVSBertConfig,
  optimizer_config: Adam_config,
  training_config: training_config,
  test_config: test_config,
  additional_kwargs: {},
};

{
  t5_executor_config: T5ExecutorConfig,
  da_executor_config: DATransformerExecutorConfig,
  test_config: test_config,
  da_transformer_model_config: DATransformerConfig,
  t5_model_config: T5ModelConfig,
  nvs_bert_model_config: NVSBertConfig,
}
