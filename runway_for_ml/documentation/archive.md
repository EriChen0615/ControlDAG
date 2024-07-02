# runway_for_ml


# Project Statement

We recognize that despite the emergence of deep Learning frameworks such as `pytorch` and higher-level frameworks such as `pytorch lightning` that separates the concerns of data, model, training, inference, and testing. There are still needs for yet a higher-level framework that addresses **data preparation**, **experiments configuration**, **systematic logging**, and **working with remote GPU clusters** (e.g. HPC). These common, indispensable functionalities are often re-implemented for individual projects, which hurt reproducibility, costs precious time, and hinders effective communications between ML developers.

We introduce **Runway**, a ML framework that delivers the last-mile solution so that researchers and engineers can focus on the essentials. In particular, we aim to 
1. Provide a **configurable data processing pipeline** that is easy to inspect and manipulate.
2. Provide an **experiment configuration system** so that it is convenient to conduct experiments in different settings without changing the code.
3. Provide a **systematic logging system** that makes it easy to log results and manage experiments both locally or on online platforms (e.g. weights-and-bias)
4. Provide a set of **tools that simplifies training/testing on remote GPU clusters** (e.g. HPC/Multiple GPU training)

With *Runway*, we hope to help ML researchers and engineers focus on the essential part of machine learning - data processing, modeling, inference, training, and evaluation. Our goal is to build a robust and flexible framework that gives developers complete freedom in these essential parts, while removing the tedious book-keeping. The philosophy, if taken to the extreme, entails that every line of code should reflect a design decision, otherwise there should be a configuration or a ready-to-use tool available. 

# Workflow

## Installation

Install with pip: `pip install runway_for_ml`

Alternatively, you can add runway as a submodule for more flexibility by running the following command

```bash
git submodule add git@github.com:EriChen0615/runway_for_ml.git runway_for_ml
git commit -m "Added runway_for_ml to the project"
```

## Initialize Runway Project

Change into target directory and run `runway init` to initialize the project. This would give you the following files/folders:

```
- data (where data is stored)
- local_cache (default caching location)
- third_party (where third party code goes)
- experiments (where experiments are stored)
- configs
    - meta_config.libsonnet
    - data_config.libsonnet
    - model_config.libsonnet
- data_processing
    - data_transforms.py
    - feature_loaders.py
- executors
    - custom_executor.py
- modeling
    - custom_modeling.py
- metrics 
    - custom_metric.py
main.py
```

## Coding Up Your Project

After setting up the runway project, you are ready to code! See [Development Manual](#-Development-Manual) for detail on how to develop code and test as you go with runway.

## Training 

Runway is organized in *experiments*. Each experiment corresponds to one trained model and possibly multiple inferences/evaluations. 

To begin training, you will need to:

1. Create an experiment configuration file using `runway-experiment <exp_name>`
2. Run training using `runway-train <exp_name>`


You can initialize a new experiment from existing ones, using `runway-experiment <exp_name> --from <existing_exp_name>`. The experiment configuration file will be cloned.

The two steps can be combined into one if you just want to change a few parameters of an existing experiment: `runway-train <exp_name> --from <existing_exp_name> --opts <param=value> ...`

## Inference & Evaluation

To run **inference**, you will need to specify an existing experiment with trained models. This can be done by `runway-infer <exp_name> <infer_suffix>`, where `<infer_suffix>` will be appended to `<exp_name>` to identify an inference run. 

You can also use `--opts` to override configurations, or use `--config <config_file>` to specify the configuration file to use for inference.

By default, evaluation will be run together with the inference. If you want to run evaluation separately, you can use `runway-eval <infer_run_name> --opts ...`

## Train/Infer/Evaluate in One Command

Runway provides a helper command that combines the above steps: `runway-run <exp_name> --from <existing_exp_name> --opts ...`. This will sequentially call `runway-train` and `runway-infer`. 

# Development Manual

Runway provides a framework to separate **data**, **modeling**, **training/inference**, and **evaluation**.

## Data

### Overview

Data are read and processed in *Data Pipeline*. A *data pipeline* is a set of inter-connected *data transform*. A *data transform* is a configurable, functional (i.e., stateless) unit that takes in some data, transform and then return it. 

### Data Transform

A *data transform* is implemented as a functor that is a subclass of *runway_for_ml.BaseTransform* that implements the `setup()` and the `_call()` function. You also need to register the transform with the `@register_transform_functor` decorator so that runway can use it. A declaration example is:

```python
@register_transform_functor
class LinearizeDialogActsTransform(HFDatasetTransform):
    def setup(self, ...):
        pass

    def _call(self, data):
        ...
        return 
```

Note that 
1. the `setup()` function configures the functor
2. the `_call()` function performs the actual transformation

You can override the following methods in the functor to implement your data transform:

- `_call(self, data)`: the argument lists depends on your superclass. The data processing logic should reside here.
- `_preprocess(self, data)`: preprocess data for transform. Can be used to handle edge cases/unify interfaces etc.
- `setup(self, *args, **kwargs)`: where the functor is configured. E.g., changing transform parameters.
- `_check_input(self, data)`: optional input checking
- `_check_output(self, data)`: optional output checking
- `_apply_mapping(self, data, in_out_col_mapping)`: this enables data field selection. May be defined in superclass.


We provide a list of ready-to-use transforms. See documentation for the full list. 

### Data Pipeline

A *data pipeline* is a connection of data transforms aranged as a **Acyclic Directed Graph (DAG)**. That is, the output of the previous transform becomes the input to the next. The developer is responsible for making sure that the input/output formats agree.

The DAG of *data pipeline* is defined in the configuration file. Below is an example:

```json
transforms: {
    "input:LoadSGDData": {
      transform_name: "LoadHFDataset",
      setup_kwargs: {
        dataset_path: "gem",
        dataset_name: "schema_guided_dialog",
      },
    },
    "process:Linearize": {
      input_node: "input:LoadSGDData",
      transform_name: "LinearizeDialogActsTransform",
      setup_kwargs: {
        linearizer_class: "SGD_TemplateGuidedLinearizer",
        schema_paths: [
          "data/schemas/train/schema.json",
          "data/schemas/test/schema.json",
          "data/schemas/dev/schema.json",
        ],
        sgd_data_dir: "data/dstc8-schema-guided-dialogue",
        template_dir: "data/utterance_templates"
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
    "output:T5-Tokenize": {
      input_node: "process:Linearize",
      transform_name: "HFDatasetTokenizeTransform",
      setup_kwargs: {
        rename_col_dict: {
          "target_input_ids": "labels",
          "target_attention_mask": "output_mask",
          "_linearized_input_ids": "input_ids",
          "_linearized_attention_mask": "attention_mask",
        },
        tokenizer_config: T5TokenizerConfig,
        tokenize_fields_list: ["target", "_linearized"],
      },
      regenerate: false,
      cache: true,
      inspect: true,
    },
  },
```

Each item in the `transform` dictonary define a node in the DAG, the important fields are:

1. The key: name of the node. Can be referenced to get data
2. `transform_name`: the name of the functor
3. `setup_kwargs`: the keyword arguments to be passed into the `setup()` function
4. `input_node`: the name of input node whose output would become the input to this node.
5. `regenerate`: whether to run the transform without using the cache
6. `cache`: whether to cache the result of the run
7. `inspect`: whether to inspect the data before/after the transform (only work with debugger now)

## Modeling








