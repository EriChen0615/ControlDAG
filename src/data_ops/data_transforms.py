import torch
from runway_for_ml.utils.global_variables import register_to, DataTransform_Registry
from runway_for_ml.data_module.data_transforms import HFDatasetTransform, BaseTransform, register_transform_functor, keep_ds_columns
from runway_for_ml.utils.util import get_tokenizer
from easydict import EasyDict
from transformers import AutoTokenizer
from typing import Dict
from datasets import load_dataset, Dataset
from .linearizer import (
    SGD_NaiveLinearizer, 
    SGD_PaperNaiveLinearizer, 
    SGD_SchemaGuidedLinearizer, 
    SGD_TemplateGuidedLinearizer,
    SGD_SchemaGuidedWithServiceLinearizer,
    SGD_SepNaiveLinearizer,
    SGD_CopyNaiveLinearizer,
    SGD_CopySchemaLinearizer,
)
from typing import List
import sys
sys.path.append('third_party')
from google_nlg.ser import get_ser_slots
from pprint import pprint
from tqdm import tqdm
import string
import json
import os
import numpy as np
from sklearn.linear_model import LinearRegression


# @register_to(DataTransform_Registry)
# def LinearizeDialogActs(
#     in_features: dict={},
#     out_features: dict=[]
#     ) -> EasyDict:
#     '''
#     Linearize fields directly
#     '''
#     output = EasyDict()
    

# @register_to(DataTransform_Registry)
# @single_feature_row_transform
# def NaiveLinearizeDialogActs(
#     dialog_act,
#     ):
#     pass
def linearize_with_fn(input_field, output_field, linear_fn):
    def linear_wrapped(example):
        linearized_dict = {output_field: linear_fn(example[input_field])}
        return linearized_dict
    return linear_wrapped

def tokenize_function(tokenizer, field, **kwargs):
    def tokenize_function_wrapped(example):
        return tokenizer.batch_encode_plus(example[field], **kwargs)
    return tokenize_function_wrapped

def _encode_intent(actions, pad_to=20, pad_value=-1):
    intent_idx = []
    for action in actions:
        if action['slot'] != '':
            intent_idx.append(action['act'])
    return np.pad(intent_idx, (1, pad_to-len(intent_idx)-1), constant_values=pad_value).astype(int)

def encode_intent(example):
    return {'intent_idx': _encode_intent(example['dialog_acts'])}

def batch_encode_intent(examples):
    batch_intent_idx = []
    for actions in examples['dialog_acts']:
        batch_intent_idx.append(_encode_intent(actions, pad_to=20))
    return {'intent_idx': batch_intent_idx}

@register_transform_functor
class LinearizeDialogActsTransform(HFDatasetTransform):
    def setup(self, linearizer_class, schema_paths=None, sgd_data_dir=None, template_dir=None):
        dataset = load_dataset('GEM/schema_guided_dialog', split='train')
        self.act_id2name_map = {i : n for i, n in enumerate(dataset.info.features['dialog_acts'][0]['act'].names)}
        self.linearizer_class = linearizer_class
        self.schema_paths = schema_paths
        self.sgd_data_dir = sgd_data_dir
        self.template_dir = template_dir
        if self.linearizer_class == 'SGD_NaiveLinearizer':
            self.linearizer = SGD_NaiveLinearizer(self.act_id2name_map)
        elif self.linearizer_class == 'SGD_PaperNaiveLinearizer':
            self.linearizer = SGD_PaperNaiveLinearizer(self.act_id2name_map)
        elif self.linearizer_class == 'SGD_SchemaGuidedLinearizer':
            self.linearizer = SGD_SchemaGuidedLinearizer(self.act_id2name_map, self.schema_paths)
        elif self.linearizer_class == 'SGD_TemplateGuidedLinearizer':
            self.linearizer = SGD_TemplateGuidedLinearizer(self.act_id2name_map, self.sgd_data_dir, self.template_dir)
        elif self.linearizer_class == 'SGD_SchemaGuidedWithServiceLinearizer':
            self.linearizer = SGD_SchemaGuidedWithServiceLinearizer(self.act_id2name_map, self.schema_paths)
        elif self.linearizer_class == 'SGD_SepNaiveLinearizer':
            self.linearizer = SGD_SepNaiveLinearizer(self.act_id2name_map, separator=" && ")
        elif self.linearizer_class == 'SGD_CopyNaiveLinearizer':
            self.linearizer = SGD_CopyNaiveLinearizer(self.act_id2name_map, self.schema_paths, separator= " && ")
        elif self.linearizer_class == 'SGD_CopySchemaLinearizer':
            self.linearizer = SGD_CopySchemaLinearizer(self.act_id2name_map, self.schema_paths, separator=" && ")
    
    def _call(self, dataset: Dataset):
        result = EasyDict()
        for split in ['train', 'test', 'validation']:
            result[split] = dataset[split].map(self.linearizer)
        return result

@register_transform_functor
class FilterServicesTransform(HFDatasetTransform):
    def setup(self, services_to_keep=None):
        self.services_to_keep = services_to_keep
    
    def _call(self, dataset: Dataset):
        for split in ['train', 'test', 'validation']:
            dataset[split] = dataset[split].filter(lambda x: x['service'] in self.services_to_keep)
        return dataset
    
@register_transform_functor
class MakeVocabularySelectionTarget(HFDatasetTransform):
    """
    Make the binary target token for Vocabulary Selection
    """
    def setup(self, tokenizer_config=None):
        tokenizer = get_tokenizer(tokenizer_config)
        self.vocab_size = len(tokenizer) # base vocab_size + len(added_tokens)
        self.pad_token_id = tokenizer.pad_token_id
    
    def _make_nvs_label(self, example):
        labels = example['labels']
        nvs_labels = torch.zeros(self.vocab_size, dtype=torch.long)
        labels_set = set(labels)
        labels_set.remove(self.pad_token_id)
        nvs_labels[list(labels_set)] = 1
        nvs_labels = nvs_labels.tolist()
        example['nvs_labels'] = nvs_labels
        return {'nvs_labels': nvs_labels}

    def _call(self, dataset: Dataset):
        result = EasyDict()
        for split in ['train', 'test', 'validation']:
            result[split] = dataset[split].map(self._make_nvs_label, load_from_cache_file=False)
        return result

@register_transform_functor
class AddBOSandEOS(HFDatasetTransform):   
    def setup(self, bos_token='<s>', eos_token='</s>', fields_to_add=["target"]):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.fields_to_add = fields_to_add
    
    def _call(self, dataset: Dataset):
        def add_bos_and_eos(example):
            for field in self.fields_to_add:
                example[field] = self.bos_token+example[field]
            return example

        result = EasyDict()
        for split in ['train', 'test', 'validation']:
            result[split] = dataset[split].map(add_bos_and_eos)
        return result

@register_transform_functor
class GetForcedTokens(HFDatasetTransform):
    def setup(self, sgd_path="data/dstc8-schema-guided-dialogue/"):
        self.permissible_slots = get_ser_slots(sgd_path)
    
    def _call(self, dataset):
        def _get_and_tokenize_noncat_slots(example):
            service = example['service']
            forced_tokens = [""]
            for action in example['dialog_acts']:
                slot = action['slot']
                if slot not in self.permissible_slots[service]:
                    continue
                else:
                    forced_tokens.extend([v for v in action['values']])
            example['forced_tokens'] = forced_tokens[1:] if len(forced_tokens) > 1 else forced_tokens # remove place holder
            return example

        result = EasyDict()
        for split in ['train', 'test', 'validation']:
            result[split] = dataset[split].map(_get_and_tokenize_noncat_slots, load_from_cache_file=False)
        return result

@register_transform_functor
class TokenizeForcedTokens(HFDatasetTransform):
    def setup(self, tokenizer_config):
        self.tokenizer = get_tokenizer(tokenizer_config)
    
    def _call(self, dataset):
        def _tokenize(example):
            token_ids = self.tokenizer(example['forced_tokens'], padding='do_not_pad', add_special_tokens=False)['input_ids']
            example['forced_token_ids'] = token_ids
            return example
            
        result = {}
        for split in ['train', 'test', 'validation']:
            ds = dataset[split]
            ds = ds.map(_tokenize)
            result[split] = ds
        return result

@register_transform_functor
class InspectTokenizedSGD(HFDatasetTransform):
    def setup(self, tokenizer_config):
        self.tokenizer = get_tokenizer(tokenizer_config)
    
    def _call(self, datasets):
        test_ds = datasets['test']
        instance = test_ds[0]
        columns = list(test_ds.info.features)
        print("="*150)
        print("Processed Data Inspection".center(150))
        print("="*150)
        print("avaiable columns")
        pprint(columns)
        print("First Instance data")
        pprint({
            col: instance[col] for col in columns if not torch.is_tensor(instance[col]) and not (type(instance[col]) == list and len(instance[col]) > 10)
        })
        print("Decoded input ids:", self.tokenizer.decode(instance['input_ids']))
        
        # compute input - output length ratio
        for split in ['train', 'test', 'validation']:
            train_ds = datasets[split]
            input_target_len_ratio = []
            max_input_length = 0
            max_input_length_instance = None
            for train_instance in tqdm(train_ds, desc=f'Analyzing {split} split'):
                input_ids, labels = train_instance['input_ids'], train_instance['labels']
                # print(input_ids, labels)
                input_length = (~torch.tensor(input_ids, dtype=torch.long).eq(0)).sum()
                tgt_length = (~torch.tensor(labels, dtype=torch.long).eq(0)).sum()
                if input_length > max_input_length:
                    max_input_length = input_length
                    max_input_length_instance = train_instance
                # print(input_length, tgt_length)
                input_target_len_ratio.append(tgt_length.item()/input_length.item())
            print(f"Max target:input ratio in {split} split", max(input_target_len_ratio))
            print(f"Min target:input ratio in {split} spilt", min(input_target_len_ratio))
            print(f"Max input length in {split} split=", max_input_length)
            print(f"Max input length instance in {split} split:", max_input_length_instance)
        return datasets

@register_transform_functor
class LinearizeDARTTriplets(HFDatasetTransform):
    def setup(self, rename_col_dict, *args, **kwargs):
        return super().setup(rename_col_dict, *args, **kwargs)

@register_transform_functor
class GetAllVocabularyInGroundTruthReference(BaseTransform):
    def setup(self, vocab_file_path):
        # self.tokenizer = get_tokenizer(tokenizer_config)
        self.vocab_file_path = vocab_file_path
    
    
    def _call(self, ds, *args, **kwargs):
        train_ds = ds['train']
        test_ds = ds['test']

        def _get_all_vocab(dataset_split):
            all_vocab = set()
            sv_vocab = set()
            for instance in tqdm(dataset_split, desc='iterating split...'):
                gt_text = instance['target'].translate(str.maketrans('', '', string.punctuation))

                slot_value_text = []
                for action in instance['dialog_acts']:
                    slot_value_text.extend(action['values'])
                slot_value_text = " ".join(slot_value_text)
                slot_value_text.translate(str.maketrans('', '', string.punctuation))

                words = gt_text.split(' ')
                for word in words:
                    if not word.isnumeric(): # do not add pure numerics to vocabulary
                        all_vocab.add(word)
                
                sv_words = slot_value_text.split(' ')
                for sv_word in sv_words:
                    if not sv_word.isnumeric():
                        sv_vocab.add(sv_word)
                        
            for punct in string.punctuation:
                all_vocab.add(punct)
                sv_vocab.add(punct)
            return all_vocab, sv_vocab

        train_vocab, _ = _get_all_vocab(train_ds)
        print("Total number of vocabulary in train split:", len(train_vocab))
        with open(os.path.join(self.vocab_file_path, 'train_vocab.json'), 'w') as f:
            json.dump([vocab for vocab in train_vocab], f)
        print(f"Vocabulary saved to {self.vocab_file_path}")
        
        test_vocab, test_sv_vocab = _get_all_vocab(test_ds)
        print("Total number of vocabulary in test split:", test_vocab)
        
        diff_vocab = test_vocab - test_sv_vocab - train_vocab
        print("Number of vocabulary not in slot values vocab and train vocab", len(diff_vocab))
        with open(os.path.join(self.vocab_file_path, 'train_test_diff_vocab.json'), 'w') as f:
            json.dump([vocab for vocab in diff_vocab], f)
        print(f"Diff Vocabulary saved to {self.vocab_file_path}")

        return ds

@register_transform_functor
class DoTargetLengthRegression(HFDatasetTransform):
    def setup(
        self,
        tokenizer_config=None,
        do_tokenize=False, 
        input_field='_linearized',
        output_field='target',
        **kwargs):
        self.do_tokenize = do_tokenize
        if self.do_tokenize and tokenizer_config is not None:
            self.tokenizer = get_tokenizer(tokenizer_config)
        self.input_field = input_field
        self.output_field = output_field
    
    def _call(self, ds, *args, **kwargs):
        input_lengths = []
        output_lengths = []
        def _get_len(example):
            example['input_len'] = [len(example[self.input_field])]
            example['output_len'] = [len(example[self.output_field])]
            return example

        # for example in tqdm(ds['train']):
        #     if self.do_tokenize:
        #         input_lengths.append(len(self.tokenizer(example[self.input_field])['input_ids']))
        #         output_lengths.append(len(self.tokenizer(example[self.output_field])['input_ids']))
        #     else:
        res_ds = ds['train'].map(_get_len, num_proc=8)
        input_lengths = list(res_ds['input_len'])
        output_lengths = list(res_ds['output_len'])
                # input_lengths.append(len(example[self.input_field]))
                # output_lengths.append(len(example[self.output_field]))
        print("input_lengths", input_lengths[:10])
        print("output_lengths", output_lengths[:10])
        input("(BREAKPOINT)")

        X = np.array(input_lengths)
        y = np.array(output_lengths)

        reg = LinearRegression().fit(X, y)
        print("R2", reg.score(X, y))
        print("Coefficient:", reg.coef_)
        print("intercept:", reg.intercept_)
        input("(BREAKPOINT)")

        test_x, test_y = [], []
        res_ds = ds['test'].map(_get_len, num_proc=8)
        test_x = list(res_ds['input_len'])
        test_y = list(res_ds['output_len'])
        
        # for example in tqdm(ds['test']):
        #     if self.do_tokenize:
        #         test_x.append(len(self.tokenizer(example[self.input_field])['input_ids']))
        #         test_y.append(len(self.tokenizer(example[self.output_field])['input_ids']))
        #     else:
        #         test_x.append(len(example[self.input_field]))
        #         test_y.append(len(example[self.output_field]))

        test_x, test_y = np.array(test_x), np.array(test_y)
        
        pred_y = reg.predict(test_x)
        rmse = np.sqrt(np.mean((test_y - pred_y)**2))
        print("Test statistics on the testset")
        print("Root Mean Square Error (RMSE):", rmse)
        print("R2", reg.score(test_x, test_y))
        input("(BREAKPOINT)")

        return ds

        