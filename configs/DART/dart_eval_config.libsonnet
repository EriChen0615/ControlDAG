local dart_eval_pipeline = {
  DataPipelineLib: 'data_modules',
  DataPipelineClass: 'DataPipeline',
  name: 'EvaluationPipeline',
  regenerate: true,
  do_inspect: true,
  // out_ops: ["output:MergeAndOutputFinalEvaluation"], 
  out_ops: ["output:FindAndSaveCases"], 
  transforms: {
    'input:GetEvaluationRecorder': {
      transform_name: 'GetEvaluationRecorder',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'input:GetGEMDARTTestReferences': {
      transform_name: 'GetGEMDARTTestReferences',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },
    'process:ComputeBLEUScoreOnDART': {
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
    'process:ComputeBLEURTScoreOnDART': {
      input_node: ['input:GetGEMDARTTestReferences', 'input:GetEvaluationRecorder'],
      transform_name: 'ComputeBLEURTScore',
      setup_kwargs: {
        checkpoint: 'third_party/bleurt/bleurt_checkpoints/BLEURT-20',
      },
      regenerate: false,
      cache: false,
    },
    // 'process:ComputeSlotErrorRate': {
    //   input_node: 'input:GetInferEvalRecorder',
    //   transform_name: 'ComputeSER',
    //   setup_kwargs: {},
    //   regenerate: false,
    //   cache: false,
    // },
    'process:ComputeDatasetLevelStats': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeAverages',
      setup_kwargs: {
        fields_to_average: ['decoded_score', 'token_recall'],
      },
      regenerate: false,
      cache: false,
    },
    'process:ComputeExactOccurRate': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeDARTExactOccurErrorRate',
      setup_kwargs: {
        exact_occur_file: 'data/dart_exact_occur_items.json',
      },
      regenerate: false,
      cache: false,
    },
    'process:ComputeNeologismRate': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeNeologismRate',
      setup_kwargs: {
        all_vocab_file: 'data/dart/total_vocab-train+test+valid-v7.json',
        lower_case: true,
        no_numeric: true,
        strip_punct: true,
      },
      regenerate: false,
      cache: false
    },
    'output:MergeAndOutputFinalEvaluation': {
      // input_node: ['process:ComputeBLEUScoreOnDART','process:ComputeDatasetLevelStats', 'process:ComputeExactOccurRate', 'process:ComputeBLEURTScoreOnDART', 'process:ComputeNeologismRate'],
      input_node: ['process:ComputeBLEUScoreOnDART', 'process:ComputeExactOccurRate', 'process:ComputeBLEURTScoreOnDART', 'process:ComputeNeologismRate'],
      // input_node: ['process:ComputeBLEUScoreOnDART', 'process:ComputeDatasetLevelStats', 'process:ComputeExactOccurRate', 'process:ComputeNeologismRate'],
      transform_name: 'MergeAllEvalRecorderAndSave',
      setup_kwargs: {},
      regenerate: true,
      cache: false,
    },
    'output:DisplayEvaluationResults': {
      input_node: 'output:MergeAndOutputFinalEvaluation',
      transform_name: 'DisplayEvalResults',
      setup_kwargs: {
        rows_to_display: 5,
        display_format: 'csv',
      },
      regenerate: true,
      cache: false,
    },
    'output:FindAndSaveCases': {
        input_node: 'output:MergeAndOutputFinalEvaluation',
        transform_name: 'AnalyzeAndSaveBadCases',
        setup_kwargs: {
            case_num: 50,
        },
        regenerate: true,
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
     'input:GetGEMTestReferences': {
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

local neologism_debug_path = {
  transforms: {
    'process:ComputeNeologismRate': {
      input_node: 'input:GetEvaluationRecorder',
      transform_name: 'ComputeNeologismRate',
      setup_kwargs: {
        all_vocab_file: 'data/dart/total_vocab-train+test+valid-v7.json',
        lower_case: true,
        no_numeric: true,
        strip_punct: true,
      },
      regenerate: false,
      cache: false
    },
    'output:MergeAndOutputFinalEvaluation': {
      input_node: ['process:ComputeNeologismRate'],
    }
  }
};

local dart_neo_pipeline = std.mergePatch(dart_eval_pipeline, neologism_debug_path);

{
  DART_eval_pipeline: dart_eval_pipeline,
  DART_neo_only_eval_pipeline: dart_neo_pipeline,
}
