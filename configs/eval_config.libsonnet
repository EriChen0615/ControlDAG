local sgd_eval_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'EvaluationPipeline',
  regenerate: true,
  do_inspect: true,
  transforms: {
    'input:GetEvaluationRecorder': {
      transform_name: 'GetEvaluationRecorder',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'input:GetGEMSGDTestReferences': {
      transform_name: 'GetGEMSGDTestReferences',
      regenerate: false,
      cache: false,
    },
    'process:ComputeBLEUScore': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'RunwayComputeBLEU',
      setup_kwargs: {
        ref_field: 'reference',
        pred_field: 'prediction',
        special_tokens_to_remove: ['<s>', '</s>'],
      },
      regenerate: false,
      cache: false,
    },
    'process:ComputeBLEURTScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetEvaluationRecorder'],
      transform_name: 'ComputeBLEURTScore',
      setup_kwargs: {
        checkpoint: 'third_party/bleurt/bleurt_checkpoints/BLEURT-20',
      },
      regenerate: false,
      cache: false,
    },
    'process:ComputeSlotErrorRate': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeSER',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'process:ComputeNeologismRate': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeNeologismRate',
      setup_kwargs: {
        all_vocab_file: 'data/dstc8-schema-guided-dialogue/all_vocab-100%+test_sv-v7.json',
        lower_case: false,
        no_numeric: true,
        strip_punct: true,
      },
      regenerate: false,
      cache: false
    },
    'process:ComputeDatasetLevelStats': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeAverages',
      setup_kwargs: {
        fields_to_average: ['decoded_score', 'token_recall'],
      },
      regenerate: false,
      cache: false,
    },
    'process:AnalyzeDecodingDetails': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'DecodingDetailsAnalysis',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'output:MergeAndOutputFinalEvaluation': {
      input_node: ['process:ComputeBLEUScore', 
      'process:ComputeBLEURTScore', 
      'process:ComputeSlotErrorRate', 
      'process:ComputeDatasetLevelStats', 
      'process:ComputeNeologismRate',
      'process:AnalyzeDecodingDetails'
      ],
    //   input_node: ['process:ComputeBLEUScore', 'process:ComputeDatasetLevelStats'],
      transform_name: 'MergeAllEvalRecorderAndSave',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'output:DisplayEvaluationResults': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'DisplayEvalResults',
      setup_kwargs: {
        rows_to_display: 5,
        display_format: 'csv',
      },
      regenerate: false,
      cache: false,
    },
    'output:FindAndSaveCases': {
        input_node: 'output:MergeAndOutputFinalEvaluation',
        transform_name: 'AnalyzeAndSaveBadCases',
        setup_kwargs: {
            case_num: 50,
        },
        cache: false,
    },
     'output:UploadToWandb': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'UploadToWandb',
      setup_kwargs: {
        log_stats_dict: true,
      },
      cache: false,
    },
  },
};

local sgd_ar_t5_eval_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'EvaluationPipeline',
  regenerate: true,
  do_inspect: true,
  transforms: {
    'input:GetInferEvalRecorder': {
      transform_name: 'GetEvaluationRecorder',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'input:GetGEMSGDTestReferences': {
      transform_name: 'GetGEMSGDTestReferences',
      regenerate: false,
      cache: false,
    },
    'process:ComputeBLEUScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
      transform_name: 'ComputeBLEU_V2',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'process:ComputeBLEURTScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
      transform_name: 'ComputeBLEURTScore',
      setup_kwargs: {
        checkpoint: 'third_party/bleurt/bleurt_checkpoints/BLEURT-20',
      },
      regenerate: false,
      cache: false,
    },
    'process:ComputeSlotErrorRate': {
      input_node: 'input:GetInferEvalRecorder',
      transform_name: 'ComputeSER',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'process:ComputeNeologismRate': {
      input_node: 'input:GetInferEvalRecorder',
      transform_name: 'ComputeNeologismRate',
      setup_kwargs: {
        all_vocab_file: 'data/dstc8-schema-guided-dialogue/all_vocab-100%+test_sv-v7.json',
        lower_case: false,
        no_numeric: true,
        strip_punct: true,
      },
      regenerate: false,
      cache: false
    },
    'process:ComputeDatasetLevelStats': {
      input_node: 'input:GetInferEvalRecorder',
      transform_name: 'ComputeAverages',
      setup_kwargs: {
        // fields_to_average: ['decoded_score', 'token_recall'],
        fields_to_average: [],
      },
      regenerate: false,
      cache: false,
    },
    'output:MergeAndOutputFinalEvaluation': {
      input_node: ['process:ComputeBLEUScore', 'process:ComputeBLEURTScore', 'process:ComputeSlotErrorRate', 'process:ComputeDatasetLevelStats', 'process:ComputeNeologismRate'],
      transform_name: 'MergeAllEvalRecorderAndSave',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'output:DisplayEvaluationResults': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'DisplayEvalResults',
      setup_kwargs: {
        rows_to_display: 5,
        display_format: 'csv',
      },
      regenerate: false,
      cache: false,
    },
    'output:FindAndSaveCases': {
        input_node: 'output:MergeAndOutputFinalEvaluation',
        transform_name: 'AnalyzeAndSaveBadCases',
        setup_kwargs: {
            case_num: 50,
        },
        cache: false,
    },
  },
};

local SGDBadCasePipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'EvaluationPipeline',
  regenerate: true,
  do_inspect: true,
  transforms: {
    'input:GetInferEvalRecorder': {
      transform_name: 'GetEvaluationRecorder',
      setup_kwargs: {
        eval_record_name: 'merged-test-evaluation',
      },
      regenerate: false,
      cache: false,
    },
    'output:FindAndSaveCasees': {
        input_node: 'input:GetInferEvalRecorder',
        transform_name: 'AnalyzeAndSaveBadCases',
        setup_kwargs: {
            case_num: 50,
        },
        cache: false,
    },
  },
};

local sgd_bleu_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'EvaluationPipeline',
  regenerate: true,
  do_inspect: true,
  transforms: {
    'input:GetInferEvalRecorder': {
      transform_name: 'GetEvaluationRecorder',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
     'input:GetGEMSGDTestReferences': {
      transform_name: 'GetGEMSGDTestReferences',
      regenerate: false,
      cache: false,
    },
    'process:ComputeBLEUScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
      transform_name: 'ComputeBLEU_V2',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'output:MergeAndOutputFinalEvaluation': {
      input_node: ['process:ComputeBLEUScore'],
    //   input_node: ['process:ComputeBLEUScore', 'process:ComputeDatasetLevelStats'],
      transform_name: 'MergeAllEvalRecorderAndSave',
      setup_kwargs: {
        eval_record_name: 'bleu-details'
      },
      regenerate: false,
      cache: false,
    },
    'output:DisplayEvaluationResults': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'DisplayEvalResults',
      setup_kwargs: {
        rows_to_display: 5,
        display_format: 'csv',
      },
      regenerate: false,
      cache: false,
    },
  },
};

local sgd_neologism_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'EvaluationPipeline',
  regenerate: true,
  do_inspect: true,
  transforms: {
    'input:GetInferEvalRecorder': {
      transform_name: 'GetEvaluationRecorder',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'process:ComputeNeologismRate': {
      input_node: 'input:GetInferEvalRecorder',
      transform_name: 'ComputeNeologismRate',
      setup_kwargs: {
        all_vocab_file: 'data/dstc8-schema-guided-dialogue/all_vocab-100%+test_sv-v7.json',
        lower_case: false,
        no_numeric: true,
        strip_punct: true,
      },
      regenerate: false,
      cache: false
    },
    'output:MergeAndOutputFinalEvaluation': {
      input_node: ['process:ComputeNeologismRate'],
    //   input_node: ['process:ComputeBLEUScore', 'process:ComputeDatasetLevelStats'],
      transform_name: 'MergeAllEvalRecorderAndSave',
      setup_kwargs: {
        eval_record_name: 'neologism-details'
      },
      regenerate: false,
      cache: false,
    },
    'output:DisplayEvaluationResults': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'DisplayEvalResults',
      setup_kwargs: {
        rows_to_display: 5,
        display_format: 'csv',
      },
      regenerate: false,
      cache: false,
    },
    
  },
};

local validation_patch = sgd_eval_pipeline {
  out_ops: ["output:UploadToWandb"],
  transforms: {
      "output:MergeAndOutputFinalEvaluation": {
        input_node: ['process:ComputeBLEUScore', 
        'process:ComputeSlotErrorRate', 
        'process:ComputeNeologismRate',
        'process:AnalyzeDecodingDetails'
        ],
      //   input_node: ['process:ComputeBLEUScore', 'process:ComputeDatasetLevelStats'],
        transform_name: 'MergeAllEvalRecorderAndSave',
        setup_kwargs: {},
        regenerate: false,
        cache: false,
    },
    'process:ComputeNeologismRate': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeNeologismRate',
      setup_kwargs: {
        all_vocab_file: 'data/dstc8-schema-guided-dialogue/all_vocab-100%+test_sv-v7.json',
        lower_case: false,
        no_numeric: true,
        strip_punct: true,
        save_to_file: false,
      },
      regenerate: false,
      cache: false
    },
    'output:UploadToWandb': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'UploadToWandb',
      setup_kwargs: {
        log_stats_dict: true,
        wandb_tab_name: 'val',
      },
      cache: false,
    },
  }
};

local sgd_valid_eval_pipeline = std.mergePatch(sgd_eval_pipeline, validation_patch);

{
  SGD_eval_pipeline: sgd_eval_pipeline,
  sgd_valid_eval_pipeline: sgd_valid_eval_pipeline,
  SGD_ar_eval_pipeline: sgd_ar_t5_eval_pipeline,
  SGD_case_analysis_pipeline: SGDBadCasePipeline,
  SGD_bleu_pipeline: sgd_bleu_pipeline,
  SGD_neologism_pipeline: sgd_neologism_pipeline,
}
