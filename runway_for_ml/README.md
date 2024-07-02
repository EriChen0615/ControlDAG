# runway_for_ml


# Overview

**Runway** is a ML framework built upon `pytorch-lightning` that delivers the last-mile solution so that researchers and engineers can focus on the essentials in ML research. The key features are:

1. A configurable functional **data processing pipeline** that is easy to inspect, use, and extend.
2. An **experiment configuration system** to conduct experiments in different settings without changing the code.
3. A **systematic logging system** that makes it easy to log results and manage experiments both locally or on online platforms (e.g. weights-and-bias)
4. A set of tools that simplifies **training/testing on remote GPU clusters** (e.g. HPC/Multiple GPU training)

With *Runway*, we hope to help ML researchers and engineers focus on the essential part of machine learning - data processing, modeling, inference, training, and evaluation. Our goal is to build a robust and flexible framework that gives developers complete freedom in these essential parts, while removing the tedious book-keeping. 

# Runway delivers research-ready ML pipeline

![The pipeline defined by Runway](assets/figures/runway_pipeline.png)

Runway organizes research ML pipeline into four stages:

1. Data Preprocessing
2. Training
3. Testing / Inference
4. Evaluation

You can define and configure each stage in the configuration file (a jsonnet file), and use the compositionality of jsonnet to modularize your config. 

## Data Preprocessing

In this stage, we preprocess our dataset for training and testing. The preprocessing is defined as a directed acyclic graph (i.e., graph with directional edges and no loops), where each node is a functional transform that takes some data and return the processed data.

![Node-level illustration](assets/figures/runway_datapipeline-Node%20Definition.png)

A node in the data pipeline has four important fields that need to be defined in the configuration file:

- **node_name**: the unique identifier of the node. Declared as key
- **input_node**: the node from which this node takes data from.
- **transform_name**: name of the data processing functor class in your code.
- **setup_kwargs**: key-value arguments to be passed into the `.setup()` function when the functor is initialized. 

Except for the first node (with name `load:<node_name>`), all other nodes will take the output of the `input_node` as its input. The node will set up (by calling `setup()`) and call the functor specified (i.e., a callable object, initialized from a class with `__call__` defined) to process the data. 

A pipeline is defined by a dictionary of node declaration, following the format:

```json
{
  "transforms": {
    "input:NameOfNode": { # name of the node 
      "input_node": "name of input node", 
      "transform_name": "name of your functor", 
      "setup_kwargs": { # used to setup the functor
        "arg_name1": "value1",
        "arg_name2": "value2",
      },
      "regenerate": false, # whether to re-run the transform, regardless of whether cache can be read
      "cache": true, # whether to save the data to cache
      "inspect": true # whether to get information printed for debugging or sanity checking
    }
  }
}
```

## Training and Testing

Training and Testing are handled by `Executor`s. An `Executor` is just a subclass of pytorch-lightning's `LightningModule`, where we define:

1. How to make the train/test/validation dataloaders
2. How to perform train/test/validation steps
3. What to do when train/test/validation ends, etc. Checkout the [LightningModule documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) 

## Experiment Management

Runway manages ML research in terms of experiments. An experiment should contain the model checkpoints, as well as all the tests which uses those checkpoints. Runway keeps your experiments organized locally, in the folder structure like following:

```
- experiments
    - <exp_name1>_V0
    - <exp_name1>_V1
        - train
          - lightning_logs
              - Version_XXX
                  - checkpoints
                      - ... ckpt files
        - test-<test_name1>
          - test_cases.csv
          - metrics.csv
        - test-<test_name2>
        ...
    - <exp_name2>_V0
    ...
```

Runway provides an automatic versioning system so that experiments with the same name are differentiated with different versions. This is handy during prototyping, but we recommend adopting explicit naming conventions to identify experiments.

## Evaluation

Evaluation takes the model"s output, run it through the evaluation pipeline to get various metrics and scores. 

Evaluation can be run separately, or automatically after training is done.


# How to Use

## Installation

<!-- Install with pip: `pip install runway_for_ml` #TODO -->

Add runway as a submodule for more flexibility by running the following command

```bash
git submodule add git@github.com:EriChen0615/runway_for_ml.git runway_for_ml
git commit -m "Added runway_for_ml to the project"
```

## Initialize Runway Project

To obtain the skeleton of a Runway project:
1. Change into the root directory of your project (i.e., root of git)
2. (Unix) run `bash runway_for_ml/init_project.sh` to initialize the project. This would give you the following folders & files:

```
- cache (default caching location)
- data (where data is stored)
- third_party (where third party code goes)
- experiments (where experiment results, including checkpoints and logs are stored)
- configs (files for configuring experiments)
    - meta_config.libsonnet
    - data_config.libsonnet
    - model_config.libsonnet
    - example.jsonnet (example file)
- src (Your source code)
    main.py (entry point to the program)
    - data_ops (where custom data transforms are defined)
        - custom_op1.py
        - custom_op2.py 
        ...
    - executors (where custom LightningModule subclasses specifying training/testing/validating are defined)
        - custom_executor1.py
        - custom_executor2.py
    - custom_folders...
    ...
```

## Data Preprocessing 

### Writing codes for data ops (data transforms) to preprocess data

You should define your data transform functor under `src/data_ops/`. To define a functor that can be used by runway, you need to:

1. Define the class for the functor, inherit one of the runway transform base classes, listed [here](#runway-data-transform-base-classes).
2. Decorate the class with `@register_transform_functor`
3. Implement `setup()` and `_call()` functions. `setup()` allows you to configurate the transform, and `_call()` is the actual transform

An example is given below:

```python
@register_transform_functor
class FilterServicesTransform(HFDatasetTransform):
    def setup(self, services_to_keep=None):
        self.services_to_keep = services_to_keep
    
    def _call(self, dataset: Dataset):
        for split in ['train', 'test', 'validation']:
            dataset[split] = dataset[split].filter(lambda x: x['service'] in self.services_to_keep)
        return dataset
```

### Define the data pipeline in config file

A *data pipeline* is a connection of data transforms aranged as a **Acyclic Directed Graph (DAG)**. That is, the output of the previous transform becomes the input to the next. The developer is responsible for making sure that the input/output formats agree.

The DAG of *data pipeline* is defined in the jsonnet configuration file. Below is an example:

```json
 {
  "data_pipeline": 
    "name": "GEMSGDDataPipeline",
    "regenerate": false,
    {
      "transforms": {
      "input:LoadSGDData": {
        "transform_name": "LoadHFDataset",
        "setup_kwargs": {
          "dataset_path": "gem",
          "dataset_name": "schema_guided_dialog",
        },
      },
      "process:Linearize": {
        "input_node": "input:LoadSGDData",
        "transform_name": "LinearizeDialogActsTransform",
        "setup_kwargs": {
          "linearizer_class": "SGD_TemplateGuidedLinearizer",
          "schema_paths": [
            "data/schemas/train/schema.json",
            "data/schemas/test/schema.json",
            "data/schemas/dev/schema.json",
          ],
          "sgd_data_dir": "data/dstc8-schema-guided-dialogue",
          "template_dir": "data/utterance_templates"
        },
        "regenerate": false,
        "cache": true,
        "inspect": true,
      },
      "output:T5-Tokenize": {
        "input_node": "process:Linearize",
        "transform_name": "HFDatasetTokenizeTransform",
        "setup_kwargs": {
          "rename_col_dict": {
            "target_input_ids": "labels",
            "target_attention_mask": "output_mask",
            "_linearized_input_ids": "input_ids",
            "_linearized_attention_mask": "attention_mask",
          },
          "tokenizer_config": T5TokenizerConfig,
          "tokenize_fields_list": ["target", "_linearized"],
        },
        "regenerate": false,
        "cache": true,
        "inspect": true,
      },
      "output:easy_SGD_Weather_1": {
        "input_node": "output:T5-Tokenize",
        "transform_name": "FilterServicesTransform",
        "setup_kwargs": {
          "services_to_keep": ["Weather_1"],
        },
        "regenerate": true,
        "cache": true,
      }
    }
  }
}, 
```


Each item in the `transform` dictonary define a node in the DAG, the important fields are:

1. The key: name of the node, in the format of `[input|process|output]:<node_name>` to indicate its role. Can be referenced to get data. 
2. `transform_name`: the name of the functor
3. `setup_kwargs`: the keyword arguments to be passed into the `setup()` function
4. `input_node`: the name of input node whose output would become the input to this node.
5. `regenerate`: whether to run the transform without using the cache
6. `cache`: whether to cache the result of the run
7. `inspect`: whether to inspect the data before/after the transform (only work with debugger now)

### Running the data pipeline

You can run the data pipeline in the commandline.

```bash
python src/main.py \
    --experiment_name "test_run" \
    --config "configs/test_run.jsonnet" \
    --mode "prepare_data" \
```

For use of CLI, refer to [detailed manual of command line](#command-line-manual)

## Training & Testing 

### Coding up Executors

An executor must implement the following functions:

- `configure_optimizers()`: return the optimizer and the scheduler
- `setup()`: create self.train_dataset, self.test_dataset and self.val_dataset available
- `training_step()`
- `test_step()`
- `validation_step()`

Optionally, it can implement/overwrite:

- `train_dataloader()`
- `test_dataloader()`
- `val_dataloader()`
- `prepare_data()`
- other functions defined in `LightningModule` [Documentation here](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html)


### Training

Run `main.py` and pass `--mode "train"` to start training.

```bash
python src/main.py \
    --experiment_name $EXPERIMENT_NAME \
    --config "configs/da-t5-bos.jsonnet" \
    --mode "train" \
    --opts \
    meta.logger_enable="[\"tensorboard\", \"wandb\"]" \
    train.batch_size=8 \
    train.trainer_paras.max_epochs=10 \
    train.trainer_paras.accelerator="gpu" \
    train.trainer_paras.devices=1 \
    train.trainer_paras.log_every_n_steps=50 \
    executor.init_kwargs.use_data_node=output:T5-T2G2Tokenize \
    executor.model_config.use_pretrained_base=False \
    executor.model_config.use_pretrained_encoder=False \
    executor.model_config.base_model_class=NAR_T5 \
```

> You can also use `--opts` to override configurations, or use `--config <config_file>` to specify the configuration file to use for inference.


## Inference & Evaluation

To run **inference**, you will need to specify the following:
- `experiment_name`: experiment name from which the model was trained (excluding version)
- `mode` = "test"
- `test_suffix`: A descriptive suffix. The results will be saved to a folder named as `test-<test_suffix>` under the same experiment. 
- `exp_version`: version number
- `test.checkpoint_name`: name of the checkpoint (only the .ckpt filename)

Example for testing.
```bash
# Test NVS Bert
python src/main.py \
    --experiment_name "NVSBert-SGD-p=100-k=200-b=8-lr=6e-3" \
    --config "configs/experiments/nvs-bert.jsonnet" \
    --mode "test" \
    --opts \
    test_suffix="ep=4" \
    exp_version="1" \
    meta.logger_enable=["csv"] \
    test.checkpoint_name="epoch=4-step=103115.ckpt" \
    test.batch_size=64 \
    test.trainer_paras.accelerator="gpu" \
    test.trainer_paras.devices=1
```

To run **evaluation**, you will need to specify the following:
- `experiment_name`: experiment name from which the model was trained (excluding version)
- `mode` = "eval"
- `test_suffix`: A descriptive suffix. The results will be saved to a folder named as `test-<test_suffix>` under the same experiment. 
- `exp_version`: version number
- `test.checkpoint_name`: name of the checkpoint (only the .ckpt filename)

Example for evaluation.
```bash
python src/main.py \
    --experiment_name "test_run-b8" \
    --config "configs/test_run.jsonnet" \
    --mode "eval" \
    --opts \
    test_suffix="ep=4;beam=4;p=0.1" \
    exp_version="0"
```


# Appendix

## Runway data transform base classes

## Command line manual

## Built-in data transforms

