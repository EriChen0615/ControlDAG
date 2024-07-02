# from .data_modules import DataPipeline
from .data_ops.data_pipeline import DataPipeline
from .configs.configuration import (
    MetaConfig, 
    DataPipelineConfig, 
    # ModelConfig, 
    # LoggingConfig, 
    # TrainingConfig, 
    # ValidationConfig, 
    # TestingConfig
)
import os
from .utils.config_system import read_config
from .utils.mixin_utils import extend_instance
from .data_ops.inspectors import DataPipelineInspector
from easydict import EasyDict


# meta_config = MetaConfig()
# dp_config = DataPipelineConfig() # data pipeline config
# next_dp_config = DataPipelineConfig() # next data pipeline config

# CONFIG_FILE = os.path.join('configs', 'exp_configs', 'example_experiment_config.jsonnet')

def initialize_config(config_file):
    config_dict = read_config(config_file)
    return config_dict 
    # meta_config = MetaConfig.from_config(config_dict)  
    # dp_config = DataPipelineConfig.from_config(config_dict, meta_config)
    # executor_config = c
    # return (
    #     config_dict,
    #     meta_config,
    #     dp_config,
    # )

def prepare_data(dp_config: DataPipelineConfig):
    # DataPipelineClass = getattr(globals()[dp_config.DataPipelineLib], dp_config.DataPipelineClass)
    # data_pipeline = DataPipelineClass(dp_config)
    data_pipeline = DataPipeline(dp_config)
    if dp_config.do_inspect:
        extend_instance(data_pipeline, DataPipelineInspector)
        data_pipeline.setup_inspector(dp_config.inspector_config)
    return data_pipeline.get_data(['output:T5-Tokenize'], explode=True)
    # processed_data = data_pipeline.run()
    # return processed_data


if __name__ == '__main__':
    # config_dict = initialize_config(CONFIG_FILE)
    # processed_data = prepare_data(config_dict)
    # train_dataloader = processed_data.train_dataloader()
    # print(next(iter(train_dataloader)))
    pass