import sys
sys.path.append('.')

from runway_for_ml.utils import config_system as rw_conf
from runway_for_ml.configs import configuration as rw_cfg
from runway_for_ml.data_module.data_pipeline import DataPipeline
from runway_for_ml.utils.global_variables import Executor_Registry
from runway_for_ml.experiment import RunwayExperiment
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from easydict import EasyDict
from pprint import pprint
import argparse

def parse_args_sys(args_list=None):
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
   
    arg_parser.add_argument('--DATA_FOLDER', type=str, default='', help='The path to data.')
    arg_parser.add_argument('--EXPERIMENT_FOLDER', type=str, default='', help='The path to save experiments.')
    
    arg_parser.add_argument('--mode', type=str, default='', help='train/test')
    arg_parser.add_argument('--reset', action='store_true', default=False, help='Reset the corresponding folder under the experiment_name')
    
    arg_parser.add_argument('--experiment_name', type=str, default='', help='Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.')
    arg_parser.add_argument("--tags", nargs='*', default=[], help="Add tags to the wandb logger")
    arg_parser.add_argument('--modules', type=str, nargs="+", default=[], help='Select modules for models. See training scripts for examples.')
    arg_parser.add_argument('--log_prediction_tables', action='store_true', default=False, help='Log prediction tables.')

    # ===== Testing Configuration ===== #
    arg_parser.add_argument('--test_batch_size', type=int, default=-1)
    arg_parser.add_argument('--test_evaluation_name', type=str, default="")
    
    
    arg_parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    
    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    return args

# def parse_sys_args():
#     arg_parser = argparse.ArgumentParser(description="")
#     arg_parser.add_argument(
#         '--config',
#         metavar='config_json_file',
#         default='None',
#         help='The Configuration file in json format'
#     )
#     arg_parser.add_argument('--experiment_name', type=str, default='', help='Experiment will be saved under /path/to/EXPERIMENT_FOLDER/$experiment_name$.')
#     arg_parser.add_argument('--from_experiment', type=str, default='', help="The Experiment name from which the new experiment inherits/overwrites config")
#     arg_parser.add_argument('--mode', type=str, default='prepare_data', help='prepare_data/train/test')
#     arg_parser.add_argument(
#         "--opts",
#         help="Modify config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
    
#     sys_args = arg_parser.parse_args()
#     return sys_args


# def _process_sys_args(config_dict, sys_args):
#     for key in vars(sys_args):
#         if key == 'opts': continue
#         value = getattr(sys_args, key)
#         config_dict[key] = value
#     _process_optional_args(config_dict, sys_args.opts)
        
# def _process_optional_args(config_dict, opts):
#     if opts is None: return
#     for opt in opts:
#         splited = opt.split('=')
#         path, value = splited[0], '='.join(splited[1:])
#         try:
#             value = eval(value)
#         except:
#             value = str(value)
#             print('input value {} is not a number, parse to string.')
#         config_key_list = path.split('.')
#         item = config_dict
#         for key in config_key_list[:-1]:
#             # assert key in item, f"Optional args error: {opt} does not exists. Error with key={key}"
#             if key not in item:
#                 item[key] = EasyDict() 
#             item = item[key]
#         item[config_key_list[-1]] = value
            

# def parse_config(config_path, sys_args):
#     config_dict = rw_conf.get_config_from_json(config_path)
#     _process_sys_args(config_dict, sys_args)
#     return config_dict    

def add_custom_sys_args(arg_parser):
    return arg_parser

def prepare_data_main(config_dict):
    meta_config = rw_cfg.MetaConfig.from_config(config_dict)
    # dp_config = rw_cfg.DataPipelineConfig.from_config(config_dict, meta_config)
    dp_config = config_dict.data_pipeline
    dp = DataPipeline(dp_config, global_config=config_dict)
    dp_config_dict = config_dict.data_pipeline

    output_data = {}
    if 'out_ops' in dp_config_dict:
        output_data = dp.get_data(dp_config_dict['out_ops'])
    else:
        output_data = dp.apply_transforms()
    print("Data Prepared!")
    print(output_data)

    
def train_main(config_dict):
    print("Runway Training...")
    rw_experiment = RunwayExperiment(config_dict)
    rw_experiment.train()

def test_main(config_dict):
    print("Runway Testing...")
    rw_experiment = RunwayExperiment(config_dict)
    rw_experiment.test()

def eval_main(config_dict):
    print("Runway Evaluating...")
    rw_experiment = RunwayExperiment(config_dict)
    rw_experiment.eval()

if __name__ == '__main__':
    print("Runway main started.")
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser = rw_conf.add_runway_sys_args(arg_parser)
    arg_parser = add_custom_sys_args(arg_parser)
    sys_args = arg_parser.parse_args()

    config_dict = rw_conf.parse_config(sys_args.config, sys_args)
    print("Configuration Loaded.")
    pprint(config_dict)

    rw_conf.import_user_modules()
    print("User modules imported")
    mode = sys_args.mode
    if mode == 'prepare_data':
        prepare_data_main(config_dict)
    elif mode == 'train':
        train_main(config_dict)
    elif mode == 'test':
        test_main(config_dict)
    elif mode == 'eval':
        eval_main(config_dict)




    