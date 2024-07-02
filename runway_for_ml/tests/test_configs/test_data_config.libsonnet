// This is the configuration file for dataloaders. It registers what dataloaders are available to use
// For each dataloader, it also registers what dataset modules are available to obtain processed features
// All dataloader and feature loaders must be declared here for runway to work
local beans_data_pipeline = {
  name: 'BeansDataPipeline',
  regenerate: false,
  do_inspect: true,
  transforms: {
    'input:LoadBeansDataset': {
      transform_name: 'LoadBeansDataset',
      setup_kwargs: {
        dataset_name: 'beans-new',
      },
      regenerate: false,
      cache: true
    },
    'output:BeanJitterTransform': {
      input_node: 'input:LoadBeansDataset',
      transform_name: 'BeansJitterTransform',
      regenerate: false,
      cache: true,
      setup_kwargs: {
        brightness: 0.5, 
        hue: 0.2
      },
    },
  }, 
};

local controlnet_testset_v1_2_preprocess_pipeline = {
    DataPipelineLib: 'data_modules',
    DataPipelineClass: 'DataPipeline',
    name: 'ArchitectDatapipeline',
    regenerate: false,
    do_inspect: true,
    inspector_config: { 
        log_dir: 'tests/'
    },
    transforms: {
        'input:ProcessControlnetTestData': {
            transform_name: 'MakeHuggingFaceDatasetFromTo0Dataset',
            setup_kwargs: {
              dataset_base_dir: 'data/Controlnet_test-set_v_1_2_2023_05_08',
              tgt_split: 'test',
            },
            cache: true,
            regenerate: false,
        },
        'process:ParsePositiveAndNegativePrompts': {
          input_node: 'input:ProcessControlnetTestData',
          transform_name: 'ParsePrompts',
          setup_kwargs: {
            prompt_rule_file: 'data/Controlnet_test-set_v_1_2_2023_05_08/prompt_rules.json',
            prompt_ds_field: 'user_prompt',
            rename_col_dict: {
              'user_prompt_pos_parsed': 'text',
              'user_prompt_neg_parsed': 'negative_prompt',
            },
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false,
        },
        'process:ResizeControlnetTestImages': {
          input_node: 'input:ProcessControlnetTestData',
          transform_name: 'ResizeAllImages',
          setup_kwargs: {
            resolution: 512,
            fields_to_resize: ['image', 'controlnet_conditional_image'],
            respect_aspect_ratio: true,
            resize_method: 'minimal',
            splits_to_process: ['test'],
            _num_proc: 8,
          },
          cache: true,
          regenerate: false,
        },
        // 'process:TokenizePrompts': {
        //   input_node: 'process:ResizeAllImages',
        //   transform_name: 'HFDatasetTokenizeTransform',
        //   setup_kwargs: {
        //     tokenizer_config: CLIPTokenizerConfig,
        //     tokenize_fields_list: ['prompt'],
        //     rename_col_dict: {
        //       'prompt_input_ids': 'input_ids',
        //       'prompt_attention_mask': 'attention_mask'
        //     },
        //   },
        //   cache: true,
        //   regenerate: false,
        // },
        'process:MakeCannyEdgeGuideImages': {
          input_node: 'process:ResizeControlnetTestImages',
          transform_name: 'MakeCannyEdgeReferenceFromSource',
          setup_kwargs: {
            image_column: 'controlnet_conditional_image',
            low_thresh: 100,
            high_thresh: 200,
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false
        },
        'process:MakeMLSDStraightLineGuideImages': {
          input_node: 'process:ResizeControlnetTestImages',
          transform_name: 'MakeMLSDStraightLineReferenceFromSource',
          setup_kwargs: {
            mlsd_model_name: 'MobileV2_MLSD_Large',
            image_column: 'controlnet_conditional_image',
            resolution: 512,
            score_thresh: 0.10,
            dist_thresh: 20.0,
            line_thickness: 1,
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false,
        },
        'process:MakeDepthGuideImages': {
          input_node: 'process:ResizeControlnetTestImages',
          transform_name: 'DummyTransform',
          setup_kwargs: {
          },
          cache: true,
          regenerate: false,
        },
        'process:MakePillowEdgeGuideImages': {
          input_node: 'process:ResizeControlnetTestImages',
          transform_name: 'MakePillowEdgeReferenceFromSource',
          setup_kwargs: {
            image_column: 'controlnet_conditional_image',
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false,
        },
        'process:MakeLineartGuideImages': {
          input_node: 'process:ResizeControlnetTestImages',
          transform_name: 'MakeLineartReferenceFromSource',
          setup_kwargs: {
            image_column: 'controlnet_conditional_image',
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false,
        },
        'Process:MergePreprocessedControlnetTestDatasetIntoOneDataset': {
          input_node: [
            'process:MakeCannyEdgeGuideImages',
            'process:MakeMLSDStraightLineGuideImages',
            'process:MakeDepthGuideImages',
            'process:MakePillowEdgeGuideImages',
            'process:MakeLineartGuideImages',
            'process:ParsePositiveAndNegativePrompts',
          ],
          transform_name: 'MergeToDatasetFromMultipleNodes',
          setup_kwargs: {
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false,
        },
        'process:ResizeAllControlnetImages': {
          input_node: 'Process:MergePreprocessedControlnetTestDatasetIntoOneDataset',
          transform_name: 'ResizeAllImages',
          setup_kwargs: {
            resolution: 512,
            fields_to_resize: ['image', 'controlnet_conditional_image', 'lineart_image', 'midas_depth_map', 'mlsd_image', 'pillow_image'],
            respect_aspect_ratio: false,
            resize_method: 'minimal',
            _num_proc: 8,
            splits_to_process: ['test'],
          },
          cache: true,
          regenerate: false
        },
        // 'output:PreprocessImages': {
        //   input_node: 'process:MakeDepthReference',
        //   transform_name: 'PreprocessTrainImages',
        //   setup_kwargs: {
        //     image_column: 'image',
        //     random_flip: true,
        //     center_crop: true,
        //   },
        //   cache: false,
        //   regenerate: true,
        // },
        'output:MakeControlnetTestDataLoaders': {
          input_node: 'process:ResizeAllControlnetImages',
          transform_name: 'MakeControlnetTestDataloaders',
          setup_kwargs: {
            use_columns: ['pixel_values', 'input_ids'],
          },
          cache: false,
          regenerate: true
        },
        'output:ExamineControlnetTestDataLoaders': {
          input_node: 'process:ResizeAllControlnetImages',
          transform_name: 'InspectControlnetTestData', 
          setup_kwargs: {
            image_save_path: 'tmp/inspected/'
          },
          cache: false,
          regenerate: true,
        },
    },
};

{
  beans_data_pipeline: beans_data_pipeline,
}