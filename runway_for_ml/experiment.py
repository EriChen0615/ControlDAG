"""
Experiment System:
    - Manage experiment hierarchy, versioning, etc.
"""
import os
import sys
from pathlib import Path
from .utils.global_variables import Executor_Registry, DataTransform_Registry
from .utils import config_system as rw_conf
from .utils import util 
from .utils.dirs import *
from .configs import configuration as rw_cfg
from .data_module.data_pipeline import DataPipeline 


import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from .utils.seed import set_seed
from easydict import EasyDict
import json
import pandas as pd
import wandb
import glob

import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
logger = logging.getLogger(__name__)

@rank_zero_only
def reset_wandb_runs(all_runs):
    for run in all_runs:
        logger.info(f'Deleting wandb run: {run}')
        run.delete()

class RunwayExperiment:
    def __init__(self, config_dict, root_dir=None):
        self.config_dict = config_dict
        self.exp_name = config_dict.get('experiment_name', None)
        self.tag = config_dict.get('tag', None)
        self.test_suffix = config_dict.get('test_suffix')
        self.meta_conf = self.config_dict['meta']

        self.root_exp_dir = root_dir or Path(self.meta_conf['EXPERIMENT_FOLDER'])
        self.exp_full_name = None
        self.exp_dir = None

        self.rw_executor = None

        # whether to use versioning in saving experiments
        self.use_versioning = config_dict.meta.get('use_versioning', True)

        # Important: set seed!
        if config_dict.meta.seed:
            set_seed(config_dict.meta.seed)
            seed_everything(config_dict.meta.seed, workers=True)
            # sets seeds for numpy, torch and python.random.
            logger.info(f'All seeds have been set to {config_dict.meta.seed}')
        
        self.loggers = None
        
        # # make paths to directories available
        # self.next_train_ver_num = 0
        # if self.use_versioning:
        #     self._check_version_and_update_exp_dir()
        
        # # self.config_dict['exp_version'] = self.ver_num
        # self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)

        # self.train_dir = self.exp_dir / 'train'
        # self.train_log_dir = self.train_dir / 'logs'
        
        if 'exp_version' not in self.config_dict: 
            self.next_train_ver_num = 0
            self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, 0, self.tag)
            if self.use_versioning:
                self._check_version_and_update_exp_dir()
            self.ver_num = self.next_train_ver_num
            self.config_dict['exp_version'] = self.ver_num
        self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.config_dict['exp_version'], self.tag)

        self.train_dir = self.exp_dir / 'train'
        self.train_log_dir = self.train_dir / 'logs'
        self.ckpt_dir = self.train_dir / 'saved_models'
        self.test_dir = (self.exp_dir / f'test') / self.test_suffix
    

        # Save some frequently-used paths to config
        self.config_dict.train_dir = str(self.train_dir)
        self.config_dict.train_log_dir = str(self.train_log_dir)
        self.config_dict.test_dir = str(self.test_dir)
        self.config_dict.ckpt_dir = str(self.ckpt_dir)

        # init wandb loggers
        ## Delete wandb logs if wandb is enabled
        if 'wandb' in self.meta_conf['logger_enable']:
            wandb_conf = self.config_dict.meta.WANDB
            config = self.config_dict

            all_runs = wandb.Api(timeout=19).runs(path=f'{wandb_conf.entity}/{wandb_conf.project}',  filters={"config.experiment_name": config.experiment_name})
            if len(all_runs) > 0 and config.mode == 'train' and config.reset:
                delete_confirm = 'n'
                dirs = [self.exp_dir]
                # Reset all the folders
                print("You are deleting following dirs: ", dirs, "input y to continue")
                if config.args.override or config.args.get('force_reset', False): # better naming than override, without breaking existing code
                    delete_confirm = 'y'
                else:
                    delete_confirm = input()
                if delete_confirm == 'y':
                    for dir in dirs:
                        try:
                            delete_dir(dir)
                        except Exception as e:
                            print(e)
                else:
                    print("reset cancelled.")
                if config.reset and config.mode == "train" and delete_confirm == 'y':
                    reset_wandb_runs(all_runs)
            else:
                if len(all_runs) > 0:
                    wandb_conf.id=all_runs[0].id
                    wandb_conf.resume="must"
            # update the original config_dict
            self.config_dict.meta.WANDB.update(wandb_conf)

    
    def _make_exp_full_name(self, exp_name, ver_num, tag):
        full_name = f"{exp_name}"

        if self.use_versioning:
            full_name = f"{full_name}_V{ver_num}"
        
        if tag is not None:
            full_name = f"{full_name}_tag:{tag}"
        
        return full_name

    def _make_experiment_dir(self, root_exp_dir, exp_name, ver_num, tag):
        self.exp_full_name = self._make_exp_full_name(exp_name, ver_num, tag)
        return root_exp_dir / self.exp_full_name 
    
    def _check_version_and_update_exp_dir(self):
        while os.path.exists(self.exp_dir):
            self.next_train_ver_num += 1
            self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.next_train_ver_num, self.tag)

    def init_loggers(self, mode='train'):
        self.logger_enable = self.meta_conf['logger_enable']
        print("Using loggers:", self.logger_enable)
        loggers = []
        if mode == 'train':
            log_dir = self.train_log_dir
        elif mode == 'test' or mode == 'eval':
            log_dir = self.test_dir
        for logger_type in self.logger_enable:
            if logger_type == "csv":
                csv_logger = pl_loggers.CSVLogger(save_dir=log_dir)
                loggers.append(csv_logger)
            elif logger_type == "tensorboard":
                tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, sub_dir='tb_log')
                loggers.append(tb_logger)
            elif logger_type == 'wandb':
                assert "WANDB" in self.meta_conf, "WANDB configuration missing in config file, but wandb_logger is used"
                config = self.config_dict
                wandb_conf = self.meta_conf["WANDB"]

                # setup wandb
                WANDB_CACHE_DIR = wandb_conf.pop('CACHE_DIR')
                if WANDB_CACHE_DIR:
                    os.environ['WANDB_CACHE_DIR'] = WANDB_CACHE_DIR
                
                # add base_model as a tag
                wandb_conf.tags.append(self.config_dict.model_config.get('base_model', self.config_dict.model_config.get('model_version', 'NoBaseModelInfo')))
                # add modules as tags
                # wandb_conf.tags.extend(self.config_dict.model_config.modules) # not every model has .modules

                logger.info('init wandb logger with the following settings: {}'.format(wandb_conf))

                wandb_logger = pl_loggers.WandbLogger(
                    name=self.exp_full_name, config=config, **wandb_conf
                )
                loggers.append(wandb_logger)
            elif logger_type == 'metrics_history':
                from .utils.metrics_log_callback import MetricsHistoryLogger
                metrics_history_logger = MetricsHistoryLogger()
                loggers.append(metrics_history_logger)

        return loggers

    def setup_sys_logs(self, log_path):

        if not os.path.exists(log_path):
            create_dirs([log_path])
        # ====== Set Logger =====
        log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s : %(message)s (in %(pathname)s:%(lineno)d)"
        log_console_format = "[%(levelname)s] - %(name)s : %(message)s"

        # Main logger
        main_logger = logging.getLogger()
        main_logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(Formatter(log_console_format))
        from .utils.color_logging import CustomFormatter
        custom_output_formatter = CustomFormatter(custom_format=log_console_format)
        console_handler.setFormatter(custom_output_formatter)

        info_file_handler = RotatingFileHandler(os.path.join(log_path, 'info.log'), maxBytes=10 ** 6,
                                                backupCount=5)
        info_file_handler.setLevel(logging.INFO)
        info_file_handler.setFormatter(Formatter(log_file_format))

        exp_file_handler = RotatingFileHandler(os.path.join(log_path, 'debug.log'), maxBytes=10 ** 6, backupCount=5)
        exp_file_handler.setLevel(logging.DEBUG)
        exp_file_handler.setFormatter(Formatter(log_file_format))

        exp_errors_file_handler = RotatingFileHandler(os.path.join(log_path, 'error.log'), maxBytes=10 ** 6,
                                                        backupCount=5)
        exp_errors_file_handler.setLevel(logging.WARNING)
        exp_errors_file_handler.setFormatter(Formatter(log_file_format))

        main_logger.addHandler(console_handler)
        main_logger.addHandler(info_file_handler)
        main_logger.addHandler(exp_file_handler)
        main_logger.addHandler(exp_errors_file_handler)

        # setup a hook to log unhandled exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                if wandb.run is not None:
                    logger.error(f"Attempting to stop the wandb run {wandb.run}")
                    wandb.finish() # stop wandb if keyboard interrupt is raised
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                
            logger.error(f"Uncaught exception: {exc_type} --> {exc_value}", exc_info=(exc_type, exc_value, exc_traceback))
            
        sys.excepthook = handle_exception
    
    def init_executor(self, mode='train'):
        meta_config = self.config_dict.meta
        dp_config = self.config_dict.data_pipeline
        valid_eval_pipeline_config = self.config_dict.eval.get('valid_eval_pipeline_config', None)
        test_eval_pipeline_config = self.config_dict.eval.get('test_eval_pipeline_config', None)
        eval_pipeline_config = self.config_dict.eval.get('eval_pipeline_config', None)
        executor_config = self.config_dict.executor
        model_config = self.config_dict.model_config
        train_config = self.config_dict.train
        test_config = self.config_dict.test

        self.loggers = self.init_loggers(mode=mode) # initialize loggers
        print(self.loggers)

        # NOTE: Tokenizer should not by default be initialised.
        tokenizer = util.get_tokenizer(self.config_dict.tokenizer_config) if 'tokenizer_config' in self.config_dict else None

        rw_executor = None
        if mode == 'train':
            rw_executor = Executor_Registry[executor_config.ExecutorClass](
                data_pipeline_config=dp_config,
                model_config=model_config,
                mode='train',
                train_config=train_config,
                test_config=test_config,
                tokenizer=tokenizer,
                valid_eval_pipeline_config=valid_eval_pipeline_config,
                global_config=self.config_dict,
                logger=self.loggers,
                **executor_config.init_kwargs
            )
        elif mode == 'test':
            def _get_lightning_version(dir_name):
                for subdir_name in os.listdir(dir_name):
                    if os.path.isdir(dir_name / subdir_name):
                        return subdir_name
            # load_ckpt_path = self.train_dir / "lightning_logs" / _get_lightning_version(self.train_dir/"lighning_logs") / "checkpoints" / test_config['checkpoint_name']
            log_file_path = self.test_dir / 'test_case.txt'
            # print("Loading checkpoint at:", load_ckpt_path)
            print("Saving testing results to:", log_file_path)
            rw_executor = Executor_Registry[executor_config.ExecutorClass](
                data_pipeline_config=dp_config,
                model_config=model_config,
                mode='test',
                test_config=test_config,
                log_file_path=log_file_path,
                test_eval_pipeline_config=test_eval_pipeline_config,
                tokenizer=tokenizer,
                global_config=self.config_dict,
                **executor_config.init_kwargs
            )
        return rw_executor
    
    def save_config_to(self, dir_path, config_filename='config'):
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        file_path = dir_path / f'{config_filename}.json'
        with open(file_path, 'w') as f:
            json.dump(self.config_dict, f, indent=4) # Added some indents to prettify the output
        logger.info(f"Config file saved to {file_path}")



    def train(self):
        train_config = self.config_dict.train

        # self.ver_num = 0
        # self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)
        
        # if self.use_versioning:
        #     self._check_version_and_update_exp_dir()
        
        # Reset the experiment (only used for training)
        # delete_confirm = 'n'
        # config = self.config_dict
        # if config.reset and config.mode == "train":
        #     dirs = [self.exp_dir]
        #     # Reset all the folders
        #     print("You are deleting following dirs: ", dirs, "input y to continue")
        #     if config.args.override or config.args.get('force_reset', False): # better naming than override, without breaking existing code
        #         delete_confirm = 'y'
        #     else:
        #         delete_confirm = input()
        #     if delete_confirm == 'y':
        #         for dir in dirs:
        #             try:
        #                 delete_dir(dir)
        #             except Exception as e:
        #                 print(e)
        #     else:
        #         print("reset cancelled.")
        
        


        # self.config_dict['exp_version'] = self.ver_num

        # self.train_dir = self.exp_dir / 'train'
        # self.train_log_dir = self.train_dir / 'logs'
        # self.ckpt_dir = self.train_dir / 'saved_models'
        
        self.setup_sys_logs(self.train_log_dir)

        # Save some frequently-used paths to config
        # self.config_dict.train_dir = str(self.train_dir)
        # self.config_dict.log_dir = str(self.train_log_dir)
        # self.config_dict.ckpt_dir = str(self.ckpt_dir)

        ## Delete wandb logs if wandb is enabled
        # if 'wandb' in self.meta_conf['logger_enable']:
        #     wandb_conf = self.config_dict.meta.WANDB
        #     config = self.config_dict

        #     all_runs = wandb.Api(timeout=19).runs(path=f'{wandb_conf.entity}/{wandb_conf.project}',  filters={"config.experiment_name": config.experiment_name})
        #     if config.reset and config.mode == "train" and delete_confirm == 'y':
        #         reset_wandb_runs(all_runs)
        #     else:
        #         if len(all_runs) > 0:
        #             wandb_conf.id=all_runs[0].id
        #             wandb_conf.resume="must"
        #     # update the original config_dict
        #     self.config_dict.meta.WANDB.update(wandb_conf)


        self.rw_executor = self.init_executor(mode='train') 

        callback_list = []
        # Checkpoint Callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=self.ckpt_dir,
            **train_config.model_checkpoint_callback_paras
        )
        callback_list.append(checkpoint_callback)

        # Early Stopping Callback
        if train_config.model_checkpoint_callback_paras.get("monitor", None) and train_config.get('early_stop_patience', 0) > 0:
            early_stop_callback = EarlyStopping(
                monitor=train_config.model_checkpoint_callback_paras.get("monitor", None) or train_config.early_stopping_callback_paras.get("monitor", None),
                **train_config.early_stopping_callback_paras
            )
            callback_list.append(early_stop_callback)

        self.save_config_to(self.train_dir)

        # 0. Load args from config_dict
        args = train_config.get('trainer_args', self.config_dict.get('args', {})) # additive change. More descriptive naming
        
        # 1. trainer parameters specified in train configs
        trainer_paras = train_config.get('trainer_paras', {})
        additional_args = trainer_paras.copy()

        # 2. update loggers, callbacks etc.
        additional_args.update({
            "default_root_dir": self.train_dir,
            'logger': self.loggers, # already initialised in advance
            'callbacks': callback_list,
        })

        # 3. update some customized behavior
        if trainer_paras.get('val_check_interval', None) is not None:
             # this is to use global_step as the interval number: global_step * grad_accumulation = batch_idx (val_check_interval is based on batch_idx)
            additional_args['val_check_interval'] = trainer_paras.val_check_interval * trainer_paras.get("accumulate_grad_batches", 1)
        
        # trainer from args
        # trainer = Trainer.from_argparse_args(args, **additional_args)
        trainer = Trainer(**args, **additional_args)
        logger.info(f"arguments passed to trainer: {str(args)}")
        logger.info(f"additional arguments passed to trainer: {str(additional_args)}")
        
        # trainer = pl.Trainer(**train_config.get('trainer_paras', {}), default_root_dir=self.train_dir ,callbacks=callback_list)
        trainer.fit(self.rw_executor, **train_config.get('trainer_fit_paras', {}))
    
    def test(self):
        logger.info("Testing starts...")
        test_config = self.config_dict.test

        if self.use_versioning:
            assert 'exp_version' in self.config_dict, "You need to specify experiment version to run test!"
        #     self.ver_num = self.config_dict['exp_version']
        # else:
        #     self.ver_num = 0
        
        # self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)
        # self.train_dir = self.exp_dir / 'train'
        # self.ckpt_dir = self.train_dir / 'saved_models'
        # self.test_dir = (self.exp_dir / f'test') / self.test_suffix

        # Save some frequently-used paths to config
        # self.config_dict.train_dir = str(self.train_dir)
        # self.config_dict.test_dir = str(self.test_dir)
        # self.config_dict.ckpt_dir = str(self.ckpt_dir)

        self.setup_sys_logs(self.test_dir)
        
        print('test-directory:', self.test_dir)
        self.save_config_to(self.test_dir)

        # Resume wandb experiments if needed
        if 'wandb' in self.meta_conf['logger_enable']:
            wandb_conf = self.config_dict.meta.WANDB
            config = self.config_dict

            all_runs = wandb.Api(timeout=19).runs(path=f'{wandb_conf.entity}/{wandb_conf.project}',  filters={"config.experiment_name": config.experiment_name})
            if len(all_runs) > 0:
                wandb_conf.id=all_runs[0].id
                wandb_conf.resume="must"
            # update the original config_dict
            self.config_dict.meta.WANDB.update(wandb_conf)


        self.rw_executor = self.init_executor(mode='test')
        # trainer = pl.Trainer(**test_config.get('trainer_paras', {}), default_root_dir=self.test_dir)
        
        # 0. Load args from config_dict
        args = test_config.get('trainer_args', self.config_dict.get('args', {}))
        
        # 1. setup additional args
        trainer_paras = test_config.get('trainer_paras', {})
        additional_args = trainer_paras.copy()

        # 2. update loggers
        additional_args.update({
            "default_root_dir": self.test_dir,
            'logger': self.loggers, # already initialised in advance
        })

        # if args.strategy == 'ddp':
        #     from pytorch_lightning.strategies import DDPStrategy
        #     additional_args['strategy'] = DDPStrategy(find_unused_parameters=True)
        
        
        
        # trainer from args
        # trainer = Trainer.from_argparse_args(args, **additional_args)
        trainer = Trainer(**args, **additional_args) # from_argparse_args is deprecated
        logger.info(f"arguments passed to trainer: {str(args)}")
        logger.info(f"additional arguments passed to trainer: {str(additional_args)}")
        
        # Auto-find checkpoints
        logger.debug(f"ckpt_dir: {self.ckpt_dir}")
        logger.debug(f"checkpoint name:{test_config.get('checkpoint_name', '')}")
        logger.debug(f"load_model_path: {test_config.get('load_model_path', None)}")
        logger.debug(f"load_best_model: {test_config.get('load_best_model', False)}")
        checkpoint_to_load = get_checkpoint_model_path(
            saved_model_path=self.ckpt_dir,
            load_checkpoint_name=test_config.get('checkpoint_name', ''), 
            load_model_path=test_config.get('load_model_path', None), 
            load_best_model=test_config.get('load_best_model', False),
        )
        if not checkpoint_to_load:
            logger.error("No checkpoint found. Please check your config file.")
            logger.error("!!! Testing continues with untrained checkpoints (also useful when applying pre-trained checkpoints directly)")
        
        trainer.test(
            self.rw_executor, 
            ckpt_path=checkpoint_to_load if checkpoint_to_load else None
        )

        self.eval()
    
    def eval(self):
        logger.info("Evaluation starts...")
        eval_config = self.config_dict.eval
        assert 'exp_version' in self.config_dict, "You must experiment version to evaluate"
        assert 'test_suffix' in self.config_dict, "You must specify name of the test run"
        # assert 'eval_op_name' in eval_config, "You must specify name of the evaluation op in .eval"
        
        self.ver_num = self.config_dict['exp_version']
        self.exp_dir = self._make_experiment_dir(self.root_exp_dir, self.exp_name, self.ver_num, self.tag)
        self.test_dir = self.exp_dir / 'test' / f"{self.test_suffix}"

        # Save some frequently-used paths to config
        self.config_dict.train_dir = str(self.train_dir)
        self.config_dict.test_dir = str(self.test_dir)
        self.config_dict.ckpt_dir = str(self.ckpt_dir)

        self.setup_sys_logs(self.test_dir)
        
        self.save_config_to(self.test_dir, config_filename='eval_config')

        if self.loggers is None:
            self.loggers = self.init_loggers(mode='eval')

        eval_pipeline_config = eval_config['pipeline_config']

        eval_pipeline = DataPipeline(eval_pipeline_config, global_config=self.config_dict)
        eval_output = {}
        if 'out_ops' in eval_pipeline_config:
            eval_output = eval_pipeline.get_data(eval_pipeline_config['out_ops'])
        else:
            eval_output = eval_pipeline.apply_transforms()
        print("Evaluation completes!")
        print(eval_output)

        # eval_op_name = eval_config['eval_op_name']
        # eval_op_kwargs = eval_config.get('setup_kwargs', {})
        # eval_op = DataTransform_Registry[eval_op_name]()
        # eval_op.setup(**eval_op_kwargs)

        # test_df = pd.read_csv(self.test_dir / 'test_case.csv')
        # eval_res_dict = eval_op._call(test_df)

        # metric_df = eval_res_dict['metrics']
        # anno_df = eval_res_dict['annotations']

        # metric_df.to_csv(self.test_dir / 'metrics.csv')
        # anno_df.to_csv(self.test_dir / 'annotated_test_case.csv')

        # print("Saved to:", self.test_dir)
        # print("Evaluation completes!")

    
def get_checkpoint_model_path(
    saved_model_path, 
    load_checkpoint_name="", 
    load_best_model=False, 
    load_model_path=""):

    if load_model_path:
        path_save_model = load_model_path
        if not os.path.exists(path_save_model):
            raise FileNotFoundError("Model file not found: {}".format(path_save_model))
    else:
        if load_best_model:
            file_name = "best.ckpt"
        else:
            if load_checkpoint_name:
                file_name = load_checkpoint_name
            else:
                logger.error("No checkpoints are specified.")
                return "" # return empty string to indicate that no model is loaded
        
        path_save_model = os.path.join(saved_model_path, file_name)

        file_names = glob.glob(f'{saved_model_path}/*.ckpt', recursive=True)
        logger.info(f'available checkpoints: {file_names}')

        if not os.path.exists(path_save_model):
            logger.warning("No checkpoint exists from '{}'. Skipping...".format(path_save_model))
            logger.info("**First time to train**")
            return '' # return empty string to indicate that no model is loaded
        else:
            logger.info("Loading checkpoint from '{}'".format(path_save_model))
    return path_save_model