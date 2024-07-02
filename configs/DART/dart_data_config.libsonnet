// This is the configuration file for dataloaders. It registers what dataloaders are available to use
// For each dataloader, it also registers what dataset modules are available to obtain processed features
// All dataloader and feature loaders must be declared here for runway to work

// data path configuration
// local default_cache_folder = '../data/ok-vqa/cache'; // override as appropriate

// Configurations for feature loaders, define as appropriate
// local example_feature_config = { // usable in ExampleLoadFeatures function
//   train: "FILE LOCATION OF TRAINING DATA",
//   test: "FILE LOCATION OF TESTING DATA",
// };
//   data_source_class: "YOUR DATA_SOURCE CLASS NAME",
//   data_source_args: { // arguments to data_source init
//   },
//   features: [ // define the `columns` of data_source
//     {
//       feature_name: "loader_name", 
//       feature_loader: example_feature_loader,
//       splits: ["train", "test", "valid"],
//     },
//   ],
// };

local default_dataloader_args = {
  batch_size: 4,
  shuffle: false,
  sampler: null,
}; // see https://pytorch.org/docs/stable/data.html for arguments

local train_transforms = [
  {
    name: 'LoadSGDDataset',
    setup_kwargs: {

    },
    inspect: true,
  },
  {
    name: 'LinearizeDialogActsTransform',
    in_col_mapping: {},
    out_col_mapping: {},
    setup_kwargs: {
      linearizer_class: 'SGD_TemplateGuidedLinearizer',
      schema_paths: [
        'data/schemas/train/schema.json',
        'data/schemas/test/schema.json',
        'data/schemas/dev/schema.json',
      ],
      sgd_data_dir: 'data/dstc8-schema-guided-dialogue',
      template_dir: 'data/utterance_templates/'
    },
    inspect: true,
  },
];

local test_transforms = train_transforms;
local valid_transforms = train_transforms;

local T5TokenizerConfig = {
  'version_name': 't5-small',
  'class_name': 'T5TokenizerFast',
  'tokenize_kwargs': {
    'padding': 'max_length',
    'truncation': true
  },
};

local T5TokenizerWithBOSConfig = {
  'version_name': 't5-small',
  'class_name': 'T5TokenizerFast',
  'tokenize_kwargs': {
    'padding': 'max_length',
    'truncation': true
  },
  'additional_tokens': ["<s>"],
};

local T5TokenizerWithBOSandGraphTokensConfig = T5TokenizerWithBOSConfig {
  // additional_tokens: ["<s>", "<H>", "<R>", "<T>"],
};

local BertTokenizerConfig = {
  'version_name': 'bert-base-uncased',
  'class_name': 'BertTokenizerFast',
  'tokenize_kwargs': {
    'padding': 'max_length',
    'truncation': true
  },
};

local GEMDART_data_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'GEMSGDDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadGEMDartData': {
      transform_name: 'LoadHFDataset',
      setup_kwargs: {
        dataset_path: 'GEM',
        dataset_name: 'dart',
      }, 
      regenerate: false,
      cache: true
    },
    'process:LinearizeDartTriples': {
      input_node: 'input:LoadGEMDartData',
      transform_name: 'LinearizeTriples',
      setup_kwargs: {
        prompt: ''
      },
      regenerate: false,
      cache: true,
    },
    'process:TokenizeFlatTripleItems': {
      input_node: 'process:LinearizeDartTriples',
      transform_name: 'TokenizeFlatTriples',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSandGraphTokensConfig,
      },
      regenerate: false,
      cache: true,
    },
    'output:TokenizeLinearizedTriples': {
      input_node: 'process:TokenizeFlatTripleItems',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
          'flat_triples_input_ids': 'triple_token_ids',
          'flat_triples_attention_mask': 'triple_attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSandGraphTokensConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
    },
    'process:GetForcedTokensForDART': {
      input_node: 'output:TokenizeLinearizedTriples',
      transform_name: 'GetForcedTokensForDART',
      setup_kwargs: {
        exact_occur_file: 'data/dart_exact_occur_items.json',
      },
      regenerate: false,
      cache: true,
    },
    'output:DARTWithForcedTokens': {
      input_node: 'process:GetForcedTokensForDART',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSandGraphTokensConfig,
      },
      regenerate: false,
      cache: true,
    },
    'output:InspectProcessedDART': {
      input_node: 'output:DARTWithForcedTokens',
      transform_name: 'InspectDartData',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSandGraphTokensConfig,
      },
      regenerate: true,
      cache: false,
    },
  },
};


{
  GEMDART_data_pipeline: GEMDART_data_pipeline,
  t5_tokenizer_with_bos_config: T5TokenizerWithBOSConfig,
  t5_tokenizer_with_bos_and_graph_config: T5TokenizerWithBOSandGraphTokensConfig,
}