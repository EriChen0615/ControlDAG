import torch
from runway_for_ml.utils.global_variables import register_to, DataTransform_Registry
from runway_for_ml.data_module.data_transforms import HFDatasetTransform, BaseTransform, register_transform_functor, keep_ds_columns
from runway_for_ml.utils.util import get_tokenizer
from pprint import pprint
from tqdm import tqdm
import json

@register_transform_functor
class LinearizeTriples(HFDatasetTransform):
    def setup(self, rename_col_dict=None, prompt='translate Graph to English: ', *args, **kwargs):
        self.prompt = prompt
        super().setup(rename_col_dict, *args, **kwargs)
    
    def _call(self, data, *args, **kwargs):
        ds = data

        def linearize_example(example):
            linearized_text = '<s>' + self.prompt 
            flat_triples = []
            triples = example['tripleset']
            for triple in triples:
                linearized_text += f"H> {triple[0]} R> {triple[1]} T> {triple[2]} "
                flat_triples.extend([item for item in triple])
            example['_linearized'] = linearized_text
            example['flat_triples'] = flat_triples
            example['target'] = '<s>' + example['target'] # add BOS token
            return example

        for split in ['train', 'test', 'validation']:
            ds[split] = ds[split].map(linearize_example, num_proc=8)

        return ds

@register_transform_functor
class TokenizeFlatTriples(BaseTransform):
    def setup(self, tokenizer_config, *args, **kwargs):
        self.tokenizer_config = tokenizer_config
        self.tokenizer = get_tokenizer(self.tokenizer_config)
    
    def _call(self, data, *args, **kwargs):
        ds = data

        def tokenize_example(example):
            tokenized_res = self.tokenizer.batch_encode_plus(example['flat_triples'], add_special_tokens=False) # do not add BOS/EOS token
            example['flat_triples_input_ids'] = tokenized_res['input_ids']
            example['flat_triples_attention_mask'] = tokenized_res['attention_mask']
            return example
        
        for split in ['train', 'test', 'validation']:
            ds[split] = ds[split].map(tokenize_example, num_proc=8)

        return ds

@register_transform_functor
class GetForcedTokensForDART(HFDatasetTransform):
    def setup(self, exact_occur_file, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.exact_occur_file = exact_occur_file
    
    def _call(self, data):
        with open(self.exact_occur_file, 'r') as f:
            exact_occur_items = json.load(f)
            exact_occur_items = set(exact_occur_items)

        def _add_constraint_text(example):
            triples = example['tripleset']
            forced_tokens = [""]
            for triple in triples:
                head, relation, tail = triple
                head = head.replace('_', ' ')
                tail = tail.replace('_', ' ')
                if f"{relation}-SUBJECT" in exact_occur_items and head not in forced_tokens:
                    forced_tokens.append(head)
                if f"{relation}-OBJECT" in exact_occur_items and tail not in forced_tokens:
                    forced_tokens.append(tail)

            example['forced_tokens'] = forced_tokens[1:] if len(forced_tokens) > 1 else forced_tokens
            return example

        for split in self.splits_to_process:
            data[split] = data[split].map(_add_constraint_text)
        # breakpoint()
        return data

@register_transform_functor
class InspectDartData(BaseTransform):
    def setup(self, tokenizer_config, *args, **kwargs):
        self.tokenizer_config = tokenizer_config
        self.tokenizer = get_tokenizer(self.tokenizer_config)
    
    def _call(self, ds, *args, **kwargs):
        train_ds = ds['train']
        print("Train examples:", len(train_ds))
        print("Validation examples:", len(ds['validation']))
        print("Test examples:", len(ds['test']))

        train_instance = train_ds[0]
        print(train_instance)
        decoded_input = self.tokenizer.decode(train_instance['input_ids'])
        pprint(f"input: {train_instance['tripleset']}")
        pprint(f"decoded input: {decoded_input}")

         # compute input - output length ratio
        datasets = ds
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

        return ds

