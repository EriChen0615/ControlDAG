"""
This file defines the config classes. The configuration file (jsonnet) will be 
converted into these config classes before they are used by the code.

Each class should implement the from_config() method, which initialize the class
members from the loaded config dict.
"""

from dataclasses import dataclass, fields
import dataclasses
from easydict import EasyDict
from abc import ABC, abstractmethod
from typing import Dict, List
import typing as t
from pprint import pprint

class ConfigClass:
    """
    Base class for all config classes. Have additional field for flexibility
    """
    additional: Dict[str, any] = None
    
    def __init__(self, *args, **kwargs):
        self.from_config(*args, **kwargs)

    # def __post_init__(self):
    # # Loop through the fields
    #     for field in fields(self):
    #         # If there is a default and the value of the field is none we can assign a value
    #         if not isinstance(field.default, dataclasses._MISSING_TYPE) and getattr(self, field.name) is None:
    #             setattr(self, field.name, None)

    @abstractmethod
    def from_config(self, config: Dict[str, any], meta_config: Dict[str, any]=None):
        raise NotImplementedError


@dataclass
class MetaConfig(ConfigClass):
    wandb_cache_dir: str = ""
    wandb_user_name: str = ""
    wandb_project_name: str = ""
    default_cache_dir: str = ""
    data_folder: str = ""
    seed: int = 2022
    platform_type: str = "pytorch"
    cuda: int = 0
    gpu_device: int = 0

    @classmethod
    def from_config(cls: t.Type['MetaConfig'], config: Dict[str, any], meta_config: Dict[str, any]=None):
        config_dict = config['meta']
        pprint(config_dict)
        return cls(
            default_cache_dir=config_dict['default_cache_dir']
        )
    


@dataclass
class DataPipelineConfig(ConfigClass):
    DataPipelineLib: str = 'data_modules'
    DataPipelineClass: str = 'DataPipeline'
    name: str = ""
    transforms: Dict[str, Dict[str, any]] = None # [split - [key - value]]
    dataloader_args: Dict[str, Dict[str, any]] = None # [split - [arg_name - arg_value]]
    cache_dir: str = ""
    do_inspect: bool = False
    inspector_config: Dict[str, any] = None

    @classmethod
    def from_config(cls: t.Type['DataPipelineConfig'], config: Dict[str, any], meta_config: Dict[str, any], key_name: str = 'data_pipeline'):
        config_dict = config[key_name]
        return cls(
            DataPipelineLib = config_dict.get('DataPipelineLib', 'data_modules'),
            DataPipelineClass = config_dict.get('DataPipelineClass', 'DataPipeline'),
            name = config_dict['name'] if 'name' in config_dict else "DefaultDataPipeline",
            transforms = EasyDict(config_dict['transforms']),
            cache_dir = config_dict['cache_dir'] if 'cache_dir' in config_dict else meta_config.default_cache_dir,
            do_inspect = config_dict.get('do_inspect', False),
            inspector_config = config_dict.get('inspector_config', None),
        )


@dataclass
class ModelConfig(ConfigClass):
    ModelLib: str = None # [transformers or custom modules]
    ModelClass: str = None
    model_version: str = None
    checkpoint_path: str = None
    load_checkpoint_kwargs: Dict[str, any] = None
    optimizer_config: Dict[str, any] = None
    training_config: Dict[str, any] = None
    additional_kwargs: Dict[str, any] = None
    # TokenizerModelVersion: str = None
    # input_transforms: List[any] = None
    # decode_input_transforms: List[any] = None
    # output_transforms: List[any] = None

    def from_config(self, config: Dict[str, any]):
        config_dict = config.model_config
        self.ModelLib = config_dict['ModelLib']
        self.ModelClass = config_dict['ModelClass']
        self.model_version = config_dict['model_name']

        self.checkpoint_path = config_dict.get('checkpoint_path', None)
        self.load_checkpoint_kwargs = config_dict.get('loading_kwargs', {})
        self.optimizer_config = config_dict.get('optimizer_config', None)
        self.training_config = config_dict.get('training_config', None)
        self.additional_kwargs = config_dict.get('additional_kwargs', None)

@dataclass
class ExecutorConfig(ConfigClass):
    ExecutorClass: str
    init_kwargs: Dict[str, any]
    dp_config: DataPipelineConfig
    model_config: ModelConfig

    def from_config(self, config: Dict[str, any]):
        config_dict = config.executor_config
        ExecutorClass = config_dict['ExecutorClass']
        init_kwargs = config_dict.get('init_kwargs', {})
        model_config = ModelConfig.from_config(config_dict)


# @dataclass
# class LoggingConfig(ConfigClass):
#     name: str

# @dataclass
# class TrainingConfig(ConfigClass):
#     executor_class: str
#     epochs: int
#     batch_size: int
#     lr: float
#     scheduler: str
#     additional: Dict[str, any]

# @dataclass
# class ValidationConfig(ConfigClass):
#     batch_size: int
#     step_size: int
#     break_interval: int
#     additional: Dict[str, any]

# @dataclass
# class TestingConfig(ConfigClass):
#     evaluation_name: str
#     load_epoch: int
#     load_model_path: str
#     load_best_model: bool
#     batch_size: int
#     num_evaluation: int
#     additional: Dict[str, any]
#     metrics: List[Dict[str, str]]





    