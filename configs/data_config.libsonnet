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

local BertTokenizerConfig = {
  'version_name': 'bert-base-uncased',
  'class_name': 'BertTokenizerFast',
  'tokenize_kwargs': {
    'padding': 'max_length',
    'truncation': true
  },
};

local GEMSGD_data_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'GEMSGDDataPipeline',
  regenerate: false,
  do_inspect: true,
  inspector_config: {
    log_dir: 'tests/'
  },
  transforms: {
    'input:LoadSGDData': {
      transform_name: 'LoadHFDataset',
      setup_kwargs: {
        dataset_path: 'GEM',
        dataset_name: 'schema_guided_dialog',
      },
    },
    'process:Linearize': {
      input_node: 'input:LoadSGDData',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_TemplateGuidedLinearizer',
        schema_paths: [
          'data/schemas/train/schema.json',
          'data/schemas/test/schema.json',
          'data/schemas/dev/schema.json',
        ],
        sgd_data_dir: 'data/dstc8-schema-guided-dialogue',
        template_dir: 'data/utterance_templates'
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'output:T5-Tokenize': {
      input_node: 'process:Linearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:NaiveLinearize': {
      input_node: 'input:LoadSGDData',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_NaiveLinearizer',
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
     'process:SchemaGuidedLinearize': {
      input_node: 'input:LoadSGDData',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_SchemaGuidedLinearizer',
        schema_paths: [
          'data/schemas/train/schema.json',
          'data/schemas/test/schema.json',
          'data/schemas/dev/schema.json',
        ],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'output:BertTokenizeNaive': {
      input_node: 'process:NaiveLinearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: BertTokenizerConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'output:BertTokenizeSchemaGuided': {
      input_node: 'process:SchemaGuidedLinearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: BertTokenizerConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'output:MakeVocabularySelectionTarget': {
      input_node: 'output:BertTokenizeNaive',
      transform_name: 'MakeVocabularySelectionTarget',
      setup_kwargs: {
        tokenizer_config: BertTokenizerConfig
      },
      regenerate: false,
      cache: true,
      inspect: true
    },
    'output:SchemaGuided_MakeVocabularySelectionTarget': {
      input_node: 'output:BertTokenizeSchemaGuided',
      transform_name: 'MakeVocabularySelectionTarget',
      setup_kwargs: {
        tokenizer_config: BertTokenizerConfig
      },
      regenerate: false,
      cache: true,
      inspect: true
    },
    'output:easy_SGD_Weather_1': {
      input_node: 'output:T5-Tokenize',
      transform_name: 'FilterServicesTransform',
      setup_kwargs: {
        'services_to_keep': ['Weather_1'],
      },
      regenerate: false,
      cache: true,
    },
    'output:easy_SGD_Weather_1_BertTokenize': {
      input_node: 'output:MakeVocabularySelectionTarget',
      transform_name: 'FilterServicesTransform',
      setup_kwargs: {
        'services_to_keep': ['Weather_1'],
      },
      regenerate: false,
      cache: true,
    },
    'output:inspect_SGD': {
      input_node: 'output:T5-Tokenize',
      transform_name: 'InspectSGDDataset',
      setup_kwargs: {},
      regenerate: false,
      cache: true,
    },
  }, 
};


local GEMSGD_exp_data_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'GEMSGDDataPipeline',
  regenerate: false,
  do_inspect: true,
  inspector_config: {
    log_dir: 'tests/'
  },
  transforms: {
    'input:LoadSGDData': {
      transform_name: 'LoadHFDataset',
      setup_kwargs: {
        dataset_path: 'GEM',
        dataset_name: 'schema_guided_dialog',
      },
      regenerate: false,
      cache: true,
    },
    'process:AddBOS': {
      input_node: 'input:LoadSGDData',
      transform_name: 'AddBOSandEOS',
      setup_kwargs: {
        bos_token: '',
        eos_token: '',
      },
      regenerate: false,
      cache: true,
    },
    'process:T2G2Linearize': {
      input_node: 'process:AddBOS',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_TemplateGuidedLinearizer',
        schema_paths: [
          'data/schemas/train/schema.json',
          'data/schemas/test/schema.json',
          'data/schemas/dev/schema.json',
        ],
        sgd_data_dir: 'data/dstc8-schema-guided-dialogue',
        template_dir: 'data/utterance_templates'
      },
      regenerate: false,
      cache: true,
    },
    'output:T5-T2G2Tokenize': {
      input_node: 'process:T2G2Linearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:GetForcedTokens': {
      input_node: 'output:T5-T2G2Tokenize',
      transform_name: 'GetForcedTokens',
      setup_kwargs: {},
      regenerate: false,
      cache: true
    },
    'output:T5-T2G2WithForcedTokens' : {
      input_node: 'process:GetForcedTokens',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig 
      },
      regenerate: false,
      cache: true,
    },
    'output:T5-T2G2_easy_SGD_Weather_1': {
      input_node: 'output:T5-T2G2WithForcedTokens',
      transform_name: 'FilterServicesTransform',
      setup_kwargs: {
        'services_to_keep': ['Weather_1'],
      },
      regenerate: false,
      cache: true,
    }, 
    # Schema Guided Linearization
    'process:SchemaGuidedLinearize': {
      input_node: 'process:AddBOS',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        // linearizer_class: 'SGD_SchemaGuidedLinearizer',
        linearizer_class: 'SGD_CopySchemaLinearizer',
        schema_paths: [
          'data/schemas/train/schema.json',
          'data/schemas/test/schema.json',
          'data/schemas/dev/schema.json',
        ],
        // sgd_data_dir: 'data/dstc8-schema-guided-dialogue',
        // template_dir: 'data/utterance_templates'
      },
      regenerate: false,
      cache: true,
    },
    'output:SchemaGuidedTokenize': {
      input_node: 'process:SchemaGuidedLinearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:GetForcedTokensSchemaGuided': {
      input_node: 'output:SchemaGuidedTokenize',
      transform_name: 'GetForcedTokens',
      setup_kwargs: {},
      regenerate: false,
      cache:true
    },
    'output:SchemaGuidedWithForcedTokens': {
      input_node: 'process:GetForcedTokensSchemaGuided',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: false,
      cache: true
    },
    'output:SchemaGuided_easy_SGD_Weather_1': {
      input_node: 'output:SchemaGuidedWithForcedTokens',
      transform_name: 'FilterServicesTransform', 
      setup_kwargs: {
        'services_to_keep': ['Weather_1'],
      },
      regenerate: false,
      cache: true,
    },
    'output:InspectProcessedSGDData': {
      input_node: 'output:T5-T2G2WithForcedTokens',
      transform_name: 'InspectTokenizedSGD',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: true,
      cache: false,
    },
    # Naive Linearization
    'process:NaiveLinearize': {
      input_node: 'process:AddBOS',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_PaperNaiveLinearizer'
      },
      regenerate: false,
      cache: true,
    },
    'output:NaiveLinearize': {
      input_node: 'process:NaiveLinearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:GetForcedTokensNaive':{
      input_node: 'output:NaiveLinearize',
      transform_name: 'GetForcedTokens',
      setup_kwargs: {},
      regenerate: false,
      cache:true
    },
    'output:NaiveWithForcedTokens': {
      input_node: 'process:GetForcedTokensNaive',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: false,
      cache: true
    },
    'output:InspectNaiveSGD': {
      input_node: 'output:NaiveWithForcedTokens',
      transform_name: 'InspectTokenizedSGD',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: true,
      cache: false,
    },
    'output:GetSchemaGuidedLengthRegressorForSGD': {
      input_node: 'output:SchemaGuidedWithForcedTokens',
      transform_name: 'DoTargetLengthRegression',
      setup_kwargs: {
        do_tokenize: false,
        input_field: '_linearized',
        output_field: 'target',
      },
      regenerate: true,
      cache: false,
    },
  }, 
};


local GEMSGD_data_pipeline_2308 = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'GEMSGDDataPipeline_2308',
  regenerate: false,
  do_inspect: true,
  inspector_config: {
    log_dir: 'tests/'
  },
  transforms: {
    'input:LoadSGDData': {
      transform_name: 'LoadHFDataset',
      setup_kwargs: {
        dataset_path: 'GEM',
        dataset_name: 'schema_guided_dialog',
      },
      regenerate: false,
      cache: true,
    },
    'process:AddBOS': {
      input_node: 'input:LoadSGDData',
      transform_name: 'AddBOSandEOS',
      regenerate: false,
      cache: true,
    },
    'process:T2G2Linearize': {
      input_node: 'process:AddBOS',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_TemplateGuidedLinearizer',
        schema_paths: [
          'data/schemas/train/schema.json',
          'data/schemas/test/schema.json',
          'data/schemas/dev/schema.json',
        ],
        sgd_data_dir: 'data/dstc8-schema-guided-dialogue',
        template_dir: 'data/utterance_templates'
      },
      regenerate: false,
      cache: true,
    },
    'output:T5-T2G2Tokenize': {
      input_node: 'process:T2G2Linearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:GetForcedTokens': {
      input_node: 'output:T5-T2G2Tokenize',
      transform_name: 'GetForcedTokens',
      setup_kwargs: {},
      regenerate: false,
      cache: true
    },
    'output:T5-T2G2WithForcedTokens' : {
      input_node: 'process:GetForcedTokens',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig 
      },
      regenerate: false,
      cache: true,
    },
    'output:T5-T2G2_easy_SGD_Weather_1': {
      input_node: 'output:T5-T2G2WithForcedTokens',
      transform_name: 'FilterServicesTransform',
      setup_kwargs: {
        'services_to_keep': ['Weather_1'],
      },
      regenerate: false,
      cache: true,
    }, 
    # Schema Guided Linearization
    'process:SchemaGuidedLinearize': {
      input_node: 'process:AddBOS',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_SchemaGuidedLinearizer',
        schema_paths: [
          'data/schemas/train/schema.json',
          'data/schemas/test/schema.json',
          'data/schemas/dev/schema.json',
        ],
        // sgd_data_dir: 'data/dstc8-schema-guided-dialogue',
        // template_dir: 'data/utterance_templates'
      },
      regenerate: false,
      cache: true,
    },
    'output:SchemaGuidedTokenize': {
      input_node: 'process:SchemaGuidedLinearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:GetForcedTokensSchemaGuided': {
      input_node: 'output:SchemaGuidedTokenize',
      transform_name: 'GetForcedTokens',
      setup_kwargs: {},
      regenerate: false,
      cache:true
    },
    'output:SchemaGuidedWithForcedTokens': {
      input_node: 'process:GetForcedTokensSchemaGuided',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: false,
      cache: true
    },
    'output:SchemaGuided_easy_SGD_Weather_1': {
      input_node: 'output:SchemaGuidedWithForcedTokens',
      transform_name: 'FilterServicesTransform', 
      setup_kwargs: {
        'services_to_keep': ['Weather_1'],
      },
      regenerate: false,
      cache: true,
    },
    'output:InspectProcessedSGDData': {
      input_node: 'output:T5-T2G2WithForcedTokens',
      transform_name: 'InspectTokenizedSGD',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: true,
      cache: false,
    },
    # Naive Linearization
    'process:NaiveLinearize': {
      input_node: 'process:AddBOS',
      transform_name: 'LinearizeDialogActsTransform',
      setup_kwargs: {
        linearizer_class: 'SGD_PaperNaiveLinearizer'
      },
      regenerate: false,
      cache: true,
    },
    'output:NaiveLinearize': {
      input_node: 'process:NaiveLinearize',
      transform_name: 'HFDatasetTokenizeTransform',
      setup_kwargs: {
        rename_col_dict: {
          'target_input_ids': 'labels',
          'target_attention_mask': 'output_mask',
          '_linearized_input_ids': 'input_ids',
          '_linearized_attention_mask': 'attention_mask',
        },
        tokenizer_config: T5TokenizerWithBOSConfig,
        tokenize_fields_list: ['target', '_linearized'],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    'process:GetForcedTokensNaive':{
      input_node: 'output:NaiveLinearize',
      transform_name: 'GetForcedTokens',
      setup_kwargs: {},
      regenerate: false,
      cache:true
    },
    'output:NaiveWithForcedTokens': {
      input_node: 'process:GetForcedTokensNaive',
      transform_name: 'TokenizeForcedTokens',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: false,
      cache: true
    },
    'output:InspectNaiveSGD': {
      input_node: 'output:NaiveWithForcedTokens',
      transform_name: 'InspectTokenizedSGD',
      setup_kwargs: {
        tokenizer_config: T5TokenizerWithBOSConfig
      },
      regenerate: true,
      cache: false,
    },
    'output:GetSchemaGuidedLengthRegressorForSGD': {
      input_node: 'output:SchemaGuidedWithForcedTokens',
      transform_name: 'DoTargetLengthRegression',
      setup_kwargs: {
        do_tokenize: false,
        input_field: '_linearized',
        output_field: 'target',
      },
      regenerate: true,
      cache: false,
    },
  }, 
};

local sgd_get_train_vocabulary_data_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'GEMSGDDataPipeline',
  regenerate: false,
  do_inspect: true,
  inspector_config: {
    log_dir: 'tests/'
  },
  transforms: {
    'input:LoadSGDData': {
      transform_name: 'LoadHFDataset',
      setup_kwargs: {
        dataset_path: 'GEM',
        dataset_name: 'schema_guided_dialog',
      },
      regenerate: false,
      cache: true,
    },
    'output:GetAllSGDTrainVocabulary': {
      input_node: 'input:LoadSGDData',
      transform_name: 'GetAllVocabularyInGroundTruthReference',
      setup_kwargs: {
        vocab_file_path: 'data/dstc8-schema-guided-dialogue/'
      },
      regenerate: true,
      cache: false,
    },
  }
};

{
  GEMSGD_data_pipeline: GEMSGD_data_pipeline,
  GEMSGD_exp_data_pipeline: GEMSGD_exp_data_pipeline,
  GEMSGD_get_vocab_pipeline: sgd_get_train_vocabulary_data_pipeline,
  tokenizer_config: T5TokenizerConfig,
  t5_tokenizer_config: T5TokenizerConfig,
  t5_tokenizer_with_bos_config: T5TokenizerWithBOSConfig,
  bert_tokenizer_config: BertTokenizerConfig,
}