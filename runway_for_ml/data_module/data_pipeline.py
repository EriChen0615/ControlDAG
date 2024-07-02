from abc import ABC, abstractmethod
from easydict import EasyDict
from collections import defaultdict
from ..configs.configuration import DataPipelineConfig
import torch
from typing import Union, List, Dict, Optional
from ..utils.cache_system import cache_data_to_disk, load_data_from_disk, cache_file_exists, make_cache_file_name
import os 
from tqdm import tqdm
from ..utils.global_variables import register_to, DataTransform_Registry
from .data_transforms import *
import hashlib
import json
from pathlib import Path
import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class DummyBase(object): pass
class DataPipeline(DummyBase):
    def __init__(
        self, 
        config: DataPipelineConfig,
        global_config=None,
        ):
        self.config = config

        self.name = self.config.get('name', 'default_pipeline')
        self.cache_dir = Path(global_config.meta.get('default_cache_dir', 'cache/'))
        self.cache_dir = self.cache_dir / self.name

        self.transforms = EasyDict(self.config.transforms)

        self.output_data = defaultdict(EasyDict) # container for output data after transformation
        self.output_cache = {}

        # Datapipeline also registered as a feature loader to compose more complex pipelines

        self.logger = None # placeholder for inspector

        # convenient variables
        self.input_transform_ids = [trans_id for trans_id in self.transforms]
        self.global_config = global_config # pass global config in, so that the transform knows the global setting
        self.use_dummy_data = self.global_config["use_dummy_data"]
        if self.use_dummy_data:
            self.cache_dir = self.cache_dir / "dummy" # dummy data is stored under dummy/ to distinguish them from full-volumn data
        logger.info(f"Using dummy data? {self.use_dummy_data}")

        self.cache_file_exists_dict = {}
        self.regenerate_all = self.config.get('regenerate', False)

        self.node_cache_hash_dict = {}
        for trans_id, trans_info in self.transforms.items():
            self._compute_md5_hash(trans_id, trans_info)
        # self.enable_fingerprint = self.config.get('enable_fingerprint', False)
        # self.cache_depend_on_input = self.config.get('cache_depend_on_input', True)
    
    def _compute_md5_hash(self, trans_id, trans_info):
        if trans_id in self.node_cache_hash_dict:
            return self.node_cache_hash_dict[trans_id]

        input_transform_ids = trans_info.get('input_node', [])
        in_node_hashes = []
        if type(input_transform_ids) == str:
            input_transform_ids = [input_transform_ids]
        for in_trans_id in input_transform_ids:
            in_node_hashes.append(self.node_cache_hash_dict.get(in_trans_id, self._compute_md5_hash(in_trans_id, self.transforms[in_trans_id])))

        dict_to_hash = trans_info.get('setup_kwargs', {}).copy()
        for key in list(dict_to_hash.keys()):
            if key.startswith('_'):
                del dict_to_hash[key] 
        # add dependency on input node's hashes
        dict_to_hash['input_hashes'] = sorted(in_node_hashes) # sorted. Make sure changing config order doesn't invalidate the cache

        md5_hash = hashlib.md5(json.dumps(dict_to_hash).encode('utf-8')).hexdigest()
        md5_hash = str(md5_hash)
        self.node_cache_hash_dict[trans_id] = md5_hash
        return md5_hash
        
    def _make_cache_filename(self, trans_id, trans_info):
        # dict_to_hash = trans_info.get('setup_kwargs', {}).copy()
        # for key in list(dict_to_hash.keys()):
        #     if key.startswith('_'):
        #         del dict_to_hash[key]
        # string_to_hash = trans_id + json.dumps(dict_to_hash)
        # md5_hash = hashlib.md5(string_to_hash.encode('utf-8')).hexdigest() 
        # cache_fname = f"{trans_id}-{str(md5_hash)}"
        cache_fname = f"{trans_id}-{self.node_cache_hash_dict[trans_id]}"
        return cache_fname
    
    def _read_from_cache(self, trans_id, trans_info):
        cache_file_name = self._make_cache_filename(trans_id, trans_info)
        FuncClass = DataTransform_Registry[trans_info['transform_name']]
        data = None
        if issubclass(FuncClass, HFDatasetTransform):
            data = load_data_from_disk(cache_file_name, self.cache_dir, save_format='hf')
        else:
            data = load_data_from_disk(cache_file_name, self.cache_dir)
        return data 

    def _save_to_cache(self, trans_id, trans_info, data):
        cache_file_name = self._make_cache_filename(trans_id, trans_info)
        FuncClass = DataTransform_Registry[trans_info['transform_name']]
        if issubclass(FuncClass, HFDatasetTransform):
            cache_data_to_disk(data, cache_file_name, self.cache_dir, save_format='hf')
        else:
            cache_data_to_disk(data, cache_file_name, self.cache_dir)
    
    def _check_cache_exist(self, trans_id, trans_info):
        trans_type, trans_name  = trans_id.split(':')
        FuncClass = DataTransform_Registry[trans_info['transform_name']]
        cache_file_name = self._make_cache_filename(trans_id, trans_info)
        
        if issubclass(FuncClass, HFDatasetTransform):
            cache_file_path = make_cache_file_name(cache_file_name, self.cache_dir, save_format='hf')
        else:
            cache_file_path = make_cache_file_name(cache_file_name, self.cache_dir)
        
        if cache_file_path not in self.cache_file_exists_dict:
            self.cache_file_exists_dict[cache_file_path] = cache_file_exists(cache_file_path) # cached for efficiency
        return self.cache_file_exists_dict[cache_file_path]
    
    def _check_input_nodes_cache_exists(self, input_trans_ids):
        all_exists = True
        if input_trans_ids is None:
            return all_exists
        if isinstance(input_trans_ids, List):
            for input_trans_id in input_trans_ids:
                input_trans_info = self.transforms[input_trans_id]
                all_exists = all_exists and (not input_trans_info.get('regenerate', False)) and self._check_cache_exist(input_trans_id, input_trans_info) and self._check_input_nodes_cache_exists(input_trans_info.get('input_node', None))
                if all_exists is False:
                    return all_exists
        else:
            input_trans_id = input_trans_ids
            input_trans_info = self.transforms[input_trans_id]
            all_exists = all_exists and (not input_trans_info.get('regenerate', False)) and self._check_cache_exist(input_trans_id, input_trans_info) and self._check_input_nodes_cache_exists(input_trans_info.get('input_node', None))
            if all_exists is False:
                return all_exists
        return all_exists


    def _exec_transform(self, trans_id, input_data_dict={}):
        # parse transform info
        trans_type, trans_name = trans_id.split(':')
        trans_info = self.transforms[trans_id]
        cache_file_name = self._make_cache_filename(trans_id, trans_info)
        
        # Read from cache or disk when available
        if trans_id in self.output_cache:
            print(f"Load {cache_file_name} from program cache")
            return self.output_cache[trans_id]
        # Read from disk when instructed and available
        elif not self.regenerate_all and not trans_info.get('regenerate', False) and self._check_cache_exist(trans_id, trans_info) and (trans_info.get('dont_check_input_cache', False) or self._check_input_nodes_cache_exists(trans_info.get('input_node', []))):
            logger.info(f"Load {cache_file_name} from disk cache")
            logger.info(f"Hash key: {self.node_cache_hash_dict[trans_id]}")
            outputs = self._read_from_cache(trans_id, trans_info)
            self.output_cache[trans_id] = outputs
            return outputs

        # Initialize functor
        # print(trans_info.transform_name)
        # print(DataTransform_Registry)
        func = DataTransform_Registry[trans_info.transform_name](
            use_dummy_data=self.use_dummy_data, 
            global_config=self.global_config,
            transform_id=trans_id,
            transform_hash=self.node_cache_hash_dict.get(trans_id, ""),
            cache_dir=trans_info.get('cache_dir', None),
            cache_base_dir=self.cache_dir,
        )
        func.setup(**trans_info.get("setup_kwargs", {}))

        # print(trans_info)
        # print(input_data_dict.keys())
        # Get input_data from all input nodes
        input_data = None
        if trans_id not in input_data_dict and trans_info.get('input_node', None):
            input_trans_ids = trans_info['input_node']
            if not isinstance(input_trans_ids, List):
                # for single node input
                input_trans_id = input_trans_ids
                input_data = self._exec_transform(input_trans_id, input_data_dict=input_data_dict)
            else:
                input_data = [
                    self._exec_transform(input_trans_id, input_data_dict=input_data_dict)
                    for input_trans_id in input_trans_ids
                ]
        elif trans_id in input_data_dict: # directly specify input_data using the input_data_dict dictionary
            input_data = input_data_dict[trans_id]
        else:
            pass
            # raise RuntimeError(f"input data to {input_trans_id} cannot be obtained. Pass in input_data_dict or define input_node for the transform")


        if hasattr(self, 'inspect_transform_before') and self.transforms[trans_id].get('inspect', True): # inspector function
            self.inspect_transform_before(trans_id, self.transforms[trans_id], input_data)

        logger.info(f"Node {trans_id}: \nExecute Transform: {trans_info.transform_name}")
        logger.info(f"Transform config: {trans_info}")
        logger.info(f"Hash key: {self.node_cache_hash_dict[trans_id]}")
        
        output = func(input_data) 
    
        if hasattr(self, 'inspect_transform_after') and self.transforms[trans_id].get('inspect', True): # inspector function
            self.inspect_transform_after(trans_id, self.transforms[trans_id], output)

        # Cache data if appropriate
        self.output_cache[trans_id] = output
        if trans_info.get('cache', False):
            self._save_to_cache(trans_id, trans_info, output)
        return output

    def apply_transforms(self, input_data_dict={}):
        for trans_id in self.transforms:
            trans_type, trans_name  = trans_id.split(':')
            if trans_type == 'output':
                self.output_data[trans_name] = self._exec_transform(trans_id, input_data_dict=input_data_dict) 
        return self.output_data
    
    def get_data(self, out_transforms, explode=False, input_data_dict={}):
        if explode:
            assert len(out_transforms)==1, "To explode data, only one field can be selected"
            return self._exec_transform(out_transforms[0], input_data_dict=input_data_dict)
        return EasyDict({
            out_trans: self._exec_transform(out_trans, input_data_dict=input_data_dict)
                for out_trans in out_transforms
        })
    
    def _get_topological_order(self, transforms):
        visited = set()
        queue = []

        def __topsort_node(trans_id, trans_info):
            input_transform_ids = trans_info.get('input_node', [])
            if type(input_transform_ids, str):
                input_transform_ids = [input_transform_ids]
            for in_trans_id in input_transform_ids:
                if in_trans_id not in visited:
                    __topsort_node(in_trans_id, self.transforms[in_trans_id])
            visited.add(trans_id)
            queue.append(trans_id)

        for trans_id, trans_info in self.transforms.items():
            __topsort_node(trans_id, trans_info)
        
        return queue

    
    def get_data_new(self, out_transforms, explode=False, input_data_dict={}):
        pass
    
    def _clear_all_program_cache(self):
        self.output_cache = {}
    
    def reset(self):
        self._clear_all_program_cache()
        

            