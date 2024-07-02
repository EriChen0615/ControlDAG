from easydict import EasyDict
from ..utils.global_variables import register_to, register_func_to_registry, DataTransform_Registry
from ..utils.util import get_tokenizer
from ..utils.eval_recorder import EvalRecorder
from transformers import AutoTokenizer
import transformers
import copy
import pandas as pd
from torchvision.transforms import ColorJitter, ToTensor
from tqdm import tqdm
from typing import Dict, List
from collections.abc import Iterable, Mapping
from datasets import Dataset, DatasetDict, load_dataset
import functools
from pathlib import Path
import sacrebleu
from pprint import pprint
import os
from PIL import Image
import wandb

import logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def register_transform(fn):
    register_func_to_registry(fn, DataTransform_Registry)
    def _fn_wrapper(*args, **kwargs):
        return fn(*args, **kwargs)
    return _fn_wrapper

def keep_ds_columns(ds, keep_cols):
    all_colummns = set(ds.features.keys())
    remove_cols = list(all_colummns - set(keep_cols))
    return ds.remove_columns(remove_cols)

def register_transform_functor(cls):
    register_func_to_registry(cls, DataTransform_Registry)
    return cls

class BaseTransform():
    """
    Most general functor definition
    """
    def __init__(
        self,
        *args,
        name=None,
        input_mapping: Dict=None,
        output_mapping: Dict=None,
        use_dummy_data=False,
        global_config=None,
        transform_id=None,
        transform_hash=None,
        cache_base_dir=None,
        cache_dir=None,
        **kwargs
        ):
        self.name = name or self.__class__.__name__
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.use_dummy_data = use_dummy_data
        self.global_config = global_config

        self.transform_id = transform_id
        self.transform_hash = transform_hash

        if cache_dir:
            self.cache_dir = cache_dir
        elif cache_base_dir:
            self.cache_dir = os.path.join(cache_base_dir, f"{self.transform_id}-{self.transform_hash}")
        else:
            base_dir = self.global_config['meta'].get('default_cache_dir', 'cache/')
            if self.use_dummy_data:
                base_dir = os.path.join(base_dir, 'dummy')
            self.cache_dir = os.path.join(base_dir, f"{self.transform_id}-{self.transform_hash}")

        os.makedirs(self.cache_dir, exist_ok=True)

    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)

    def __call__(self, data, *args, **kwargs):
        preprocessed_data = self._preprocess(data) # any preprocessing should be handled here
        # mapped_data = self._apply_mapping(preprocessed_data, self.input_mapping)
        self._check_input(preprocessed_data)

        # output_data = self._call(**mapped_data) if self.input_mapping else self._call(mapped_data)
        output_data = self._call(preprocessed_data)
        # output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_data)

        return output_data
        
        # _call will expand keyword arguments from data if mapping [input_col_name : output_col_name] is given
        # otherwise received whole data
    
    # def _apply_mapping(self, data, in_out_col_mapping):
    #     """
    #     IMPORTANT: when input_mapping is given, data will be transformed into EasyDict
    #     """
    #     if in_out_col_mapping is None:
    #         return data
    #     assert isinstance(data, Mapping), f"input feature mapping cannot be performed on non-Mapping type objects!"
    #     mapped_data = {}
    #     for input_col, output_col in in_out_col_mapping.items():
    #         mapped_data[output_col] = data[input_col]
    #     return EasyDict(mapped_data)



    def _check_input(self, data):
        """
        Check if the transformed can be applied on data. Override in subclasses
        No constraints by default
        """
        return True
    
    def _check_output(self, data):
        """
        Check if the transformed data fulfills certain conditions. Override in subclasses
        No constraints by default
        """
        return True
        
    
    def _preprocess(self, data):
        """
        Preprocess data for transform.
        """
        return data

    def setup(self, *args, **kwargs):
        """
        setup any reusable resources for the transformed. Will be called before __apply__()
        """
        raise NotImplementedError(f"Must implement {self.name}.setup() to be a valid transform")

    def _call(self, data, *args, **kwargs):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

class RowWiseTransform(BaseTransform):
    """
    Transform each element row-by-row
    """
    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)

    def __call__(self, data, *args, **kwargs):
        preprocesed_data = self._preprocess(data) # any preprocessing should be handled here
        self._check_input(preprocesed_data)
        for row_n, row_data in enumerate(preprocesed_data):
            mapped_data = self._apply_mapping(row_data, self.input_mapping)
            output_data = self._call(row_n, **mapped_data) if self.input_mapping else self._call(row_n, mapped_data)
            output_mapped_data = self._apply_mapping(output_data, self.output_mapping)
        self._check_output(output_mapped_data)
        return output_mapped_data

    def _call(self, row_n, row_data):
        raise NotImplementedError(f'Must implement {self.name}._call() to be a valid transform')

    def _check_input(self, data):
        return isinstance(data, Iterable)

class HFDatasetTransform(BaseTransform):
    """
    Transform using HuggingFace Dataset utility
    """
    # @classmethod 
    # def __init__subclass__(cls, **kwargs):
    #     super().__init_subclass__(*args, **kwargs)
    #     register_func_to_registry(cls.__name__, DataTransform_Registry)
    def setup(self, rename_col_dict=None, splits_to_process=['train', 'test', 'validation'], _num_proc=1, *args, **kwargs):
        """
        setup any reusable resources for the transformed. Will be called before __call__()
        For HFDataset, add rename_col_dict for renaming columns conveniently
        """
        self.rename_col_dict = rename_col_dict
        self.splits_to_process = splits_to_process
        self._num_proc = _num_proc

    def _check_input(self, data):
        return isinstance(data, Dataset) or isinstance(data, DatasetDict) or data is None
    
    def image_magic(self, img_path_fields, out_img_fields=None, batched=False, save_out_images=True):
        """A magic decorator for image processing. It does the following:
        1. Automatically convert image paths (`img_path_fields`) to PIL.Image before passing them into the decorated function.
        2. Save processed images specified (`out_image_fields`, or `img_path_fields` if unspecified) to cache_dir. `cache_dir` is specified in the transform constructor.
        3. Convert processed images back to paths before returning the result. This ensure the images are not doubly stored in the arrow files.

        Typical Usage:
         - decorate the function used in `map` of HFDatasetTransform. 
        
        Example:
        class ResizeAllImages_Native(ResizeAllImages)
            def _call(self, ds, *args, **kwargs):
                @self.image_magic(self.fields_to_resize)
                def resize_image(example):
                    for img_field in self.fields_to_resize:
                        if img_field in example.keys():
                            # Fixed: width and height are switched when using np.array()
                            img = example[img_field] #np.array(example[img_field])
                            tgt_W, tgt_H = self._compute_W_H_for_image((img.size[0], img.size[1]))
                            example[img_field] = img.resize((tgt_W, tgt_H))
                    return example
            
                @self.image_magic(self.fields_to_resize, batched=True)
                def batch_resize_image(examples):
                    for img_field in self.fields_to_resize:
                        res = []
                        orig = []
                        for img in examples[img_field]:
                            tgt_W, tgt_H = self._compute_W_H_for_image((img.size[0], img.size[1]))
                            resized_img = img.resize((tgt_W, tgt_H))
                            res.append(resized_img)
                            orig.append(img)

                        examples[img_field] = res
                    return examples
                    
                res = {} #DatasetDict()
                for split in ds:
                    if split in self.splits_to_process:
                        if not self.batched:
                            resized_ds = ds[split].map(resize_image, num_proc=self.num_proc, **self.map_kwargs)
                        else:
                            resized_ds = ds[split].map(batch_resize_image, num_proc=self.num_proc, batched=True, **self.map_kwargs)
                        res[split] = resized_ds
                    else:
                        res[split] = ds[split]
                return res
            

        :param img_path_fields: <list> of field names to be processed
        :param out_img_fields: <list> of image field names to save, defaults to None
        :param batched: <bool> whether the function used in map is batched, defaults to False
        :param save_out_images: <bool> whether the images specified by `out_image_fields` should be saved to cache_dir, defaults to True
        """
        def transform_decorator(func):
            def wrapper(example, *args, **kwargs):
                """func must return a dictionary

                :param func: _description_
                :param example: _description_
                :return: _description_
                """
                img_name_memo = None 
                img_path_memo = None
                if not batched:
                    img_name_memo = {img_path_field: example[img_path_field].split('/')[-1] for img_path_field in img_path_fields}
                    img_path_memo = {img_path_field: example[img_path_field] for img_path_field in img_path_fields}
                    for img_path in img_path_fields:
                        example[img_path] = Image.open(example[img_path])
                else: # batch operation
                    img_name_memo = {img_path_field: [pp.split('/')[-1] for pp in example[img_path_field] ] for img_path_field in img_path_fields}
                    img_path_memo = {img_path_field: example[img_path_field] for img_path_field in img_path_fields}
                    for img_path in img_path_fields:
                        example[img_path] = [Image.open(pp) for pp in example[img_path]]
                
                processed_example = func(example, *args, **kwargs)

                # PROCESS OUT IMAGES FIELDS
                name_base_field = img_path_fields[0] #TODO can be more flexible with naming

                img_out_fields = out_img_fields or img_path_fields
                if not batched:
                    for img_out_field in img_out_fields:
                        image_name = img_name_memo[name_base_field] if len(img_out_fields) == 1 else f"{img_out_field}-{img_name_memo[name_base_field]}"
                        img_save_path = os.path.join(self.cache_dir, image_name)
                        if save_out_images:
                            processed_example[img_out_field].convert('RGB').save(img_save_path)
                        processed_example[img_out_field] = img_save_path
                else:
                    for img_out_field in img_out_fields:
                        for i, img_out in enumerate(processed_example[img_out_field]):
                            image_name = img_name_memo[name_base_field][i] if len(img_out_fields) == 1 else f"{img_out_field}-{img_name_memo[name_base_field][i]}"
                            img_save_path = os.path.join(self.cache_dir, image_name)
                            if save_out_images:
                                img_out.save(img_save_path)
                            processed_example[img_out_field][i] = img_save_path
                
                # Convert in image fields from PIL.Image back to paths (only for read-only images. Output images would have already been converted to paths)
                for img_in_field in img_path_fields:
                    if img_in_field not in img_out_fields:
                        processed_example[img_in_field] = img_path_memo[img_in_field]

                # read images to fields
                return processed_example
            # do something
            return wrapper
        return transform_decorator
    
    def log_pv(page_id):
        def log_pv_decorator(func, *args, **kwargs):
            def wrapper(func, *args, **kwargs):
                return func(*args, **kwargs)
            # Do somthing
            return wrapper
        return log_pv_decorator
    
    # def _apply_mapping(self, data, in_out_col_mapping):
    #     if not in_out_col_mapping:
    #         return data
    #     if isinstance(data, DatasetDict):
    #         mapped_data = {out_col_name: data[in_col_name] for in_col_name, out_col_name in in_out_col_mapping.items()}
    #         return mapped_data
    #     else: # data is DatasetDict
    #         data = data.rename_columns(in_out_col_mapping)
    #         mapped_data = keep_ds_columns(data, list(in_out_col_mapping.values()))
    #         return mapped_data
    
def tokenize_function(tokenizer, field, **kwargs):
    def tokenize_function_wrapped(example):
        return tokenizer.batch_encode_plus(example[field], **kwargs)
    return tokenize_function_wrapped

@register_transform_functor
class HFDatasetTokenizeTransform(HFDatasetTransform):
    def setup(self, rename_col_dict, tokenizer_config: EasyDict, tokenize_fields_list: List, splits_to_process=['train', 'test', 'validation']):
        super().setup(rename_col_dict)
        self.tokenize_fields_list = tokenize_fields_list
        self.tokenizer = get_tokenizer(tokenizer_config)
        self.tokenize_kwargs = tokenizer_config.get(
            'tokenize_kwargs', 
            {
             'batched': True,
             'load_from_cache_file': False,
             'padding': 'max_length',
             'truncation': True
             }
        )
        self.splits_to_process = splits_to_process

    def _call(self, dataset):
        results = {}
        for split in ['train', 'test', 'validation']:
            # ds = dataset[split].select((i for i in range(100)))
            if split not in dataset:
                continue
            ds = dataset[split]
            for field_name in self.tokenize_fields_list:
                ds = ds\
                .map(tokenize_function(self.tokenizer, field_name, **self.tokenize_kwargs), batched=True, load_from_cache_file=False) \
                .rename_columns({
                    'input_ids': field_name+'_input_ids',
                    'attention_mask': field_name+'_attention_mask',
                })
            ds = ds.rename_columns(self.rename_col_dict)
            results[split] = ds
        return results

@register_transform_functor
class LoadHFDataset(BaseTransform):
    def setup(self, dataset_name, dataset_path=None, fields=[], load_kwargs=None):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.load_kwargs = load_kwargs or {}
        self.fields = fields
    
    def _call(self, data):
        dataset_url = None
        if self.dataset_path:
            dataset_url = f"{self.dataset_path}/{self.dataset_name}"
        else:
            dataset_url = self.dataset_name
        hf_ds = load_dataset(dataset_url, **self.load_kwargs)
        if 'split' in self.load_kwargs:
            hf_ds = DatasetDict({split_name: hf_ds[i] for i, split_name in enumerate(self.load_kwargs['split'])})

        if self.use_dummy_data:
            for ds in hf_ds:
                ds = ds[:10] 
        return hf_ds

@register_transform_functor
class SplitHFDatasetToTrainTestValidation(HFDatasetTransform):
    def setup(self, test_size, train_test_split_kwargs={}, valid_size=None):
        self.test_size = test_size
        self.valid_size = valid_size
        self.test_valid_total_size = self.test_size + self.valid_size if self.valid_size else self.test_size
        self.train_test_split_kwargs = train_test_split_kwargs
        # assert self.test_valid_total_size <= 1.0
    
    def _call(self, data, *args, **kwargs):
        train_ds = data['train']
        train_dict = train_ds.train_test_split(self.test_valid_total_size, **self.train_test_split_kwargs)
        train_ds = train_dict['train']
        test_ds = train_dict['test']
        if self.valid_size is not None:
            test_valid_dict = train_dict['test'].train_test_split(self.valid_size / self.test_valid_total_size, **self.train_test_split_kwargs)
            test_ds = test_valid_dict['train']
            valid_ds = test_valid_dict['test']

        res_dataset_dict = DatasetDict({
            'train': train_ds,
            'test': test_ds,
            'validation': valid_ds
        })
        print("Split into train/test/validation:",res_dataset_dict)
        return res_dataset_dict

@register_transform_functor
class DummyTransform(BaseTransform):
    def setup(self):
        pass

    def _call(self, data):
        return data

@register_transform_functor
class GetEvaluationRecorder(BaseTransform):
    def setup(self, base_dir=None, eval_record_name='test-evaluation', recorder_prefix='eval_recorder', file_format='json', keep_top_n=None):
        self.eval_record_name = eval_record_name
        self.recorder_prefix = recorder_prefix
        self.base_dir = base_dir or self.global_config['test_dir']
        self.file_format = file_format
        self.keep_top_n = keep_top_n
    
    def _call(self, data):
        if data is not None:
            return data # short cut for validation pipeline
        eval_recorder = EvalRecorder.load_from_disk(self.eval_record_name, self.base_dir, file_prefix=self.recorder_prefix, file_format=self.file_format)
        if self.keep_top_n:
            eval_recorder.keep_top_n(self.keep_top_n)
        return eval_recorder
    
@register_transform_functor
class MergeAllEvalRecorderAndSave(BaseTransform):
    def setup(
        self, 
        base_dir = None, 
        eval_record_name='merged-test-evaluation', 
        eval_recorder_prefix='merged',
        recorder_prefix='eval_recorder', 
        file_format='json', 
        save_recorder=True
    ):
        self.eval_record_name = eval_record_name
        self.eval_recorder_prefix = eval_recorder_prefix
        self.recorder_prefix = recorder_prefix
        self.base_dir = base_dir
        self.file_format = file_format
        self.save_recorder = save_recorder

    def _call(self, data):
        """_summary_

        :param data: _description_
        """
        eval_recorder = data[0]
        # self.base_dir = self.base_dir or str(Path(eval_recorder.save_dir).parent)
        if len(data) > 1:
            eval_recorder.merge(data[1:]) # merge all evaluation results
        if self.eval_recorder_prefix is not None:
            self.eval_record_name = f"{self.eval_recorder_prefix}-{eval_recorder.name}"
        eval_recorder.rename(self.eval_record_name)
        # eval_recorder.rename(self.eval_record_name, new_base_dir=self.base_dir)
        eval_recorder.save_to_disk(self.recorder_prefix, file_format=self.file_format)
        logger.warning(f"Evaluation recorder merged and saved to {eval_recorder.save_dir}")
        return eval_recorder
        
@register_transform_functor
class RunwayComputeBLEU(BaseTransform):
    """_summary_
    
    example config:

    'process:ComputeBLEUScore': {
      input_node: ['input:GetGEMSGDTestReferences', 'input:GetInferEvalRecorder'],
      transform_name: 'ComputeBLEU',
      setup_kwargs: {},
      regenerate: false,
      cache: false,
    },

    :param BaseTransform: _description_
    """
    def setup(self, ref_field, pred_field, inp_field=None, special_tokens_to_remove=[]):
        self.ref_field = ref_field
        self.pred_field = pred_field
        self.inp_field = inp_field
        self.special_tokens_to_remove = special_tokens_to_remove

    def _call(self, eval_recorder, *args, **kwargs):
        """data must contain keys 'eval_recorder' and 'references', other keys are optional

        :param data: _description_
        :return: _description_
        """
        refs = eval_recorder.get_sample_logs_column(self.ref_field)
        hypos = eval_recorder.get_sample_logs_column(self.pred_field)
        # print(hypos)
        # input("Press Enter to continue...")

        # Remove special tokens 
        def _remove_special_tokens(text):
            for token in self.special_tokens_to_remove:
                text = text.replace(token, "")
            return text
        refs = [_remove_special_tokens(ref).strip() for ref in refs]
        hypos = [_remove_special_tokens(hypo).strip() for hypo in hypos] # remove hypothesis

        setence_bleu_score = self.compute_sentence_bleu(hypos, refs)
        corpus_bleu_res = self.compute_corpus_bleu(hypos, refs)

        for bleu_field in ['score', 'bp', 'sys_len', 'ref_len']:
            eval_recorder.log_stats_dict({f'pred_corpus_bleu_{bleu_field}': getattr(corpus_bleu_res, bleu_field)})
        eval_recorder.log_stats_dict({'pred_corpus_bleu_without_bp': corpus_bleu_res.score / corpus_bleu_res.bp if corpus_bleu_res.bp > 0.01 else 0.0})

        eval_recorder.set_sample_logs_column('sentence_bleu', setence_bleu_score)
        eval_recorder.log_stats_dict({'avg_sentence_bleu': sum(setence_bleu_score)/len(setence_bleu_score)})

        return eval_recorder

    def compute_sentence_bleu(self, preds, refs):
        sentence_bleu_res = []
        for r, p in zip(refs, preds):
            sentence_bleu_res.append(sacrebleu.sentence_bleu(p, [r]).score)
        return sentence_bleu_res

    def compute_corpus_bleu(self, preds, refs):
        print("Compute corpus bleu:", len(preds), len(refs))
        corpus_bleu = sacrebleu.corpus_bleu(preds, [refs])
        # corpus_bleu = sacrebleu.corpus_bleu(preds, [[r] for r in refs])
        return corpus_bleu
        
@register_transform_functor
class RunwayDisplayEvalResults(BaseTransform):
    def setup(self, rows_to_display=5, display_format='csv'):
        self.rows_to_display = rows_to_display
        self.display_format = display_format
    
    def print_boarder(self):
        print("="*150)
    
    def _call(self, eval_recorder, *args, **kwargs):
        if self.display_format == 'csv':
            df = eval_recorder.get_sample_logs(data_format='csv')
            print("Available columns in sample logs:")
            pprint(df.columns.tolist())
            with pd.option_context('display.max_rows', self.rows_to_display, 'display.max_columns', None):
                print(df.head(n=self.rows_to_display))
        self.print_boarder()
        # pprint(f"Full evaluation data saved to {eval_recorder.save_dir}")
        # logger.warning(f"Full evaluation data saved to {eval_recorder.save_dir}")
        # self.print_boarder()
        print(f"Evaluation Report for {self.global_config['experiment_name']}".center(150))
        self.print_boarder()
        pprint(eval_recorder.get_stats_logs(data_format='dict'))
        return eval_recorder

@register_transform_functor
class UploadToWandb(BaseTransform):
    def setup(self, log_stats_dict=True, columns_to_log=None, prefix_to_log=None, wandb_tab_name=None):
        self.log_stats_dict = log_stats_dict
        self.columns_to_log = columns_to_log
        self.prefix_to_log = prefix_to_log
        self.wandb_tab_name = wandb_tab_name
    
    def _call(self, eval_recorder):
        if not 'wandb' in self.global_config['meta']['logger_enable']:
            print("wandb is not enabled ('wandb' not in meta.logger_enable). Pass.")
            return eval_recorder
        
        sec_name = self.wandb_tab_name or eval_recorder.name
        if self.log_stats_dict:
            stats_dict = eval_recorder.get_stats_logs()
            # wandb.log({f"{sec_name}/{k}": v for k, v in stats_dict.items()}, step=eval_recorder.get_stats_logs()['global_step'], commit=True)
            wandb.log({f"{sec_name}/{k}": v for k, v in stats_dict.items()},  commit=True)
            print(f"Eval Recorder: {sec_name} stats dict uploaded to wandb")
            print(stats_dict)
        if not (self.columns_to_log is None and self.prefix_to_log is None):
            table_data = {}
            for colname in eval_recorder.get_sample_logs().keys():
                if (self.columns_to_log is not None and colname in self.columns_to_log) or \
                (self.prefix_to_log is not None and any([colname.startswith(prefix) for prefix in self.prefix_to_log])):
                    values = eval_recorder.get_sample_logs_column(colname)
                    table_data[colname] = values
            df = pd.DataFrame(table_data)
            wandb_table = wandb.Table(data=df)
            wandb.log({f"{sec_name}/table": wandb_table}, step=eval_recorder.get_stats_logs('global_step'), commit=True)    
            print(f"Eval Recorder: {sec_name} columns {df.columns.tolist()} uploaded to wandb")
        return eval_recorder