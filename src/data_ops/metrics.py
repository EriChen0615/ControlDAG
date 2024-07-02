from runway_for_ml.data_module.data_transforms import BaseTransform, HFDatasetTransform, register_transform_functor, keep_ds_columns
from runway_for_ml.utils.util import get_tokenizer
from pprint import pprint
import pandas as pd
from datasets import load_metric, load_dataset
from abc import ABC, abstractmethod
import pandas as pd
import sys
sys.path.append('third_party')
from google_nlg.ser import get_ser_slots, example_ser
import json
from tqdm import tqdm
import torch
import sacrebleu

class Metric(ABC):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.records = {'ref':[], 'pred':[], 'tags': []}
        self.setup(*args, **kwargs)
    
    @abstractmethod
    def setup(self, *args, **kwargs):
        pass
    
    def __repr__(self):
        return f"Metric: {self.name}"
    
    def add_batch(self, references=None, predictions=None, tags=None):
        if references is None or predictions is None:
            raise ValueError("reference or predictions is None")
        self.records['ref'].extend(references)
        self.records['pred'].extend(predictions)
        self.records['tags'].extend(tags or [None for i in range(len(references))])

    def compute(self, reference=None, prediction=None, tag=None, full=True, avg=False):
        if reference or prediction or tag:
            return self.compute_row({'ref': reference, 'pred': prediction, 'tag': tag})
        else:
            self.record_df = pd.DataFrame(self.records)
            res = self.record_df.apply(self.compute_row, axis='columns')
            self.record_df[self.name] = res
            if full:
                return self.record_df
            elif not avg:
                return self.record_df[self.name]
            else:
                return self.record_df[self.name].mean()

    @abstractmethod
    def compute_row(self, row):
        pass

class SER_metric(Metric):
    '''
    ref: list dialog_acts
    pred: string 
    This applies to non-categorical values only

    '''
    def __init__(self, name='SER', data_dir="", contain_fn=None):
        super().__init__(name, data_dir=data_dir, contain_fn=contain_fn)
        
    def setup(self, data_dir="", contain_fn=None, *args, **kwargs):
        self.permissible_slots = get_ser_slots(data_dir)
        permissible_slot_set = set()
        for service, slot_names in self.permissible_slots.items():
            for slot in slot_names:
                permissible_slot_set.add(slot)
        self.permissible_slots = list(permissible_slot_set)

        self.contain_fn = contain_fn or (lambda pred, v: v in pred)
    
    def compute_row(self, row):
        '''
        Compute if there is a slot value for the current record: True: error, False: no error, None: not applicable
        '''
        ref = row['ref']
        pred = row['pred'].lower()
        tags = row['tags']
        service = tags['service']
        has_permissible = False
        for action in tags['dialog_acts']:
            slot = action['slot']
            if slot not in self.permissible_slots[service]:
                continue
            values = action['values']
            for v in values:
                has_permissible = True # it could be that the slot is permissible but no value is given. We shouldn't include these bonus
                v = v.lower()
                if not self.contain_fn(pred, v):
                    return True
        return False if has_permissible else None

def has_slot_error(turn, pred, permissible_slots):
    pred = pred.lower()
    service = turn['service'] 
    ser_applicable = False
    for action in turn['dialog_acts']:
        slot = action['slot']
        if slot not in permissible_slots[service]:
            continue
        ser_applicable = True
        values = action['values']
        for v in values:
            v = v.lower()
            if v not in pred:
                return True, ser_applicable
    return False, ser_applicable

def compute_ser(turns, preds):
    """Return sentence and corpus level SER

    Parameters
    ----------
    turns : A turn in GEM-SGD
        _description_
    preds : A list of predicted system utterance
        _description_

    Returns
    -------
    list
        turn-level SER
    float
        corpus-level SER
    """
    permissible_slots = get_ser_slots("data/dstc8-schema-guided-dialogue/")
    ser_res = []
    ser_applicable = []
    ser_err_cnt = 0
    for turn, pred in zip(turns, preds):
        ser_error, applicable = has_slot_error(turn, pred, permissible_slots=permissible_slots)
        ser_applicable.append(applicable)
        if ser_error:
            ser_err_cnt += 1
        ser_res.append(ser_error)
    return ser_res, ser_applicable, ser_err_cnt / len(ser_res), ser_err_cnt / len([1 for app in ser_applicable if app == True])

class DARTExactOccurRateEvaluator:
    def __init__(self, exact_occur_file):
        with open(exact_occur_file, 'r') as f:
            self.exact_occur_items = json.load(f)
            self.exact_occur_items = set(self.exact_occur_items)
    
    def compute_eor(self, triples, pred):
        forced_phrases = []
        for triple in triples:
            head, relation, tail = triple
            head = head.replace('_', ' ')
            tail = tail.replace('_', ' ')
            if f"{relation}-SUBJECT" in self.exact_occur_items:
                forced_phrases.append(head)
            if f"{relation}-OBJECT" in self.exact_occur_items:
                forced_phrases.append(tail)
        
        for forced_phrase in forced_phrases:
            if forced_phrase not in pred:
                return True, forced_phrases # has exact occur error
        return False, forced_phrases


@register_transform_functor
class EvaluateNLGOutput(BaseTransform):
    def setup(self, *args, **kwargs):
        self.bleu_eval = load_metric('sacrebleu')
        self.ser_eval = SER_metric(name='ser-strict', data_dir='data/dstc8-schema-guided-dialogue/')
        self.slot_error_cnt = 0
        self.permissible_slots = get_ser_slots("data/dstc8-schema-guided-dialogue/")
        self.test_dataset = load_dataset('gem', 'schema_guided_dialog', split='test')
        # self.test_dataset = self.test_dataset.filter(lambda x: x['service']=='Weather_1') #TODO only evaluate this
    
    def _call(self, data, *args, **kwargs):
        test_df = data
        test_df['dialog_acts'] = self.test_dataset['dialog_acts']
        test_df['service'] = self.test_dataset['service']
        test_df['domain'] = list(map(lambda x: x.split('_')[0], self.test_dataset['service']))
        test_df['has_slot_error'] = -1
        
        slot_error_list = []
        for idx, row in tqdm(test_df.iterrows()):
            ref = row['reference']
            pred = row['prediction']
            # turn_bleu_score = turn_bleu.compute(references=[[ref]], predictions=[pred])
            # record_df.iloc[idx]['bleu'] = turn_bleu_score['score'] extremely slow
            self.bleu_eval.add_batch(references=[[ref]], predictions=[pred])
            self.ser_eval.add_batch(references=[ref], predictions=[pred])
            slot_error = has_slot_error(self.test_dataset[idx], pred, self.permissible_slots)
            if slot_error:
                self.slot_error_cnt += 1
                slot_error_list.append(True)
            else:
                slot_error_list.append(False)

        test_df['has_slot_error'] = slot_error_list

        bleu_res = self.bleu_eval.compute()
        # ser_res = self.ser_eval.compute()

        metric_df = pd.DataFrame(
            {
                'bleu': [bleu_res['score']],
                'ser': self.slot_error_cnt / len(test_df),
            }
        )

        return {
            'metrics': metric_df,
            'annotations': test_df,
        } 

def compute_vs_recall_with_binary_encoding(preds, labels):
    """
    @parameters:
        preds: predicted binary encoding; size = [batch_size, vocab_size]
        labels: ground-truth binary encoding: size = [batch_size, vocab_size]
    @returns:
        example_recall: List. Recall at example level
        avg_recall: float. Average recall
    """
    example_recall = ((labels==preds) * (labels==1)).sum(dim=-1)/(labels).sum(dim=-1)
    avg_recall = example_recall.mean().item()
    return example_recall.tolist(), avg_recall

def compute_vs_recall_with_vocab_prob(pred_probs, labels, top_k_list=[200]):
    """
    pred_probs: probability output for each tokens; size = [batch_size, vocab_size]; DOES NOT SUM TO 1
    labels: ground-truth binary encoding: size = [batch_size, vocab_size]
    """
    res_dict = {}
    for top_k in top_k_list:
        pred_binary = _make_topk_binary_indicator_from_prob(pred_probs, top_k)
        example_recall, avg_recall = compute_vs_recall_with_binary_encoding(pred_binary, labels)
        res_dict[top_k] = {'example_recall': example_recall, 'avg_recall': avg_recall}
    return res_dict

def _make_topk_binary_indicator_from_prob(pred_probs, top_k):
    batch_size = pred_probs.shape[0]
    topk_pred_idx = torch.topk(pred_probs, top_k, dim=1)[1]
    pred_binary = torch.zeros_like(pred_probs, dtype=torch.long)
    batch_idx = torch.arange(batch_size)[:,None].repeat((1, top_k)).flatten()
    token_idx = topk_pred_idx.flatten()
    pred_binary[batch_idx, token_idx] = 1
    return pred_binary
    

@register_transform_functor
class ComputeNVSRecall(BaseTransform):
    def setup(self, tokenizer_config, top_k_list=[200], *args, **kwargs):
        self.tokenizer = get_tokenizer(tokenizer_config)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.top_k_list = top_k_list
        self.vocab_size = len(self.tokenizer)

    def _call(self, data):
        """
        Parameters
        ----------
        data : Dictionary
            [key]="preds": vocabulary selection probability (after sigmoid), size = (N, vocab_size)
            [key]="labels": ground-truth vocabulary as binary indicator vectors, size = (N, vocab_size)
        Returns
        -------
        res_dict: Dictionary
            [key]=<top_k:int>: result dictionary with k=<top_k>. Contains overall recall and case-wise recall 
        """
        labels = data['labels'] # size = (batch_size, vocab_size), binary indicator vector
        preds = data['preds'] # size = (batch_size, top_k), predicted vocabulary idx

        res_dict = compute_vs_recall_with_vocab_prob(preds, labels, top_k_list=self.top_k_list) 

        return res_dict

def compute_sentence_bleu(preds, refs):
    sentence_bleu_res = []
    for r, p in zip(refs, preds):
        sentence_bleu_res.append(sacrebleu.sentence_bleu(p, [r]).score)
    return sentence_bleu_res

def compute_corpus_bleu(preds, refs):
    corpus_bleu = sacrebleu.corpus_bleu(preds, [[r] for r in refs])
    return corpus_bleu.score

