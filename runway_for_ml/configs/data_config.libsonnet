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

local base_feature_loader = {
  name: "FEATURE LOADER name here",
  kwargs: {}, // arguments to feature_loader init
  cache_data: true,
  use_cache: true,
};

local beans_feature_loader = base_feature_loader {
  name: "LoadBeansDataset",
  kwargs: {}, // arguments to feature_loader init
  cache_data: true,
  use_cache: true,
};

// local example_data_source = {
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
    name: 'ColorJitterTransform',
    use_features: ['image'],
    out_features: ['transformed_image'], // override col1; if 'col1+', result will be appended to col1
    batched: 0,
    kwargs: {},
  },
  {
    name: 'ToTensorTransform',
    use_features: ['transformed_image'],
    out_features: ['tensor_image'],
    kwargs: {},
  },
  {
    name: 'CopyFields',
    use_features: ['image', 'labels'],
    out_features: ['image', 'labels'],
    kwargs: {},
  },
];

local test_transforms = [
  {
    name: 'CopyFields',
    use_features: ['image', 'labels'],
    out_features: ['image', 'labels'],
    kwargs: {},
  },
];

local valid_transforms = test_transforms;

local base_data_pipeline = {
  name: 'BaseDatasetPipeline',
  regenerate: true,
  dataloader_args: {
    train: default_dataloader_args {
      shuffle: true // override
    },
    test: default_dataloader_args {

    },
    valid: default_dataloader_args {

    },
  },
  in_features: [ // features used by the pipelines (MUST BE available at init)
    {
      feature_names: ['feature1', 'feature2'],
      feature_loader: base_feature_loader,
      splits: ["train", "test", "valid"], // the splits available
      use_cache: true,
    },
  ],
  transforms: {
    train: train_transforms,
    test: test_transforms,
    valid: valid_transforms,
  },
  dataloaders_use_features: {
    train: ['feature1', 'feature2'],
    test: ['feature1', 'feature2'],
    valid: ['feature1', 'feature2'],
  },
};

local beans_data_pipeline = base_data_pipeline {
  name: 'BeansDatasetPipeline',
  regenerate: false,
  in_features: [ // features used by the pipelines (MUST BE available at init)
    {
      feature_names: ['image', 'labels'],
      feature_loader: beans_feature_loader,
      splits: ["train", "test", "valid"], // the splits available
      use_cache: true,
    },
  ],

  transforms: {
    train: train_transforms,
    test: test_transforms,
    valid: valid_transforms,
  },
  dataloaders_use_features: {
    train: ['tensor_image', 'labels'],
    test: ['image', 'labels'],
    valid: ['image', 'labels'],
  },
};

local next_beans_data_pipeline = beans_data_pipeline {
  name: 'NextBeansDatasetPipeline',
  regenerate: true,
  in_features: [
    {
      feature_names: ['image', 'labels'],
      feature_loader: beans_feature_loader {
        name: 'LoadBeansDatasetPipeline',
        splits: ["train", "test", "valid"], // the splits available
        use_cache: true,
      },
      splits: ["train", "test", "valid"], // the splits available
      use_cache: true,
    },
  ],
};


{
  example_data_pipeline: beans_data_pipeline,
  next_data_pipeline: next_beans_data_pipeline,
}