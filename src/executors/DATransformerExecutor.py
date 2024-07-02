import torch
import torch.nn.functional as F
import pandas as pd
import os.path
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from models.modeling_da_transformer import DirectedAcyclicTransformer
from models.configuration_da_transformer import DirectedAcyclicTransformerConfig
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import wandb
from data_ops.metrics import compute_sentence_bleu, compute_corpus_bleu
import copy
from datasets import load_dataset
import json
from pprint import pprint
import time
from runway_for_ml.utils.util import get_tokenizer
from runway_for_ml.utils.eval_recorder import EvalRecorder

@register_executor
class DATransformerExecutor(BaseExecutor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        eval_pipeline_config=None,
        global_config=None,
        dump_dag=False,
        test_with_train_split=False,
        use_length_constraint=None, # ['orcale', 'regressor', None]
        length_regressor_coeff=None,
        torch_precision='high',
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, eval_pipeline_config=eval_pipeline_config, global_config=global_config, *args, **kwargs)
        self.use_data_node = use_data_node
        self.tokenizer = tokenizer
        if self.mode == 'test':
            self.save_decode = test_config.get('save_decode', True)
            if self.save_decode:
                if not os.path.exists(os.path.dirname(log_file_path)):
                    os.makedirs(os.path.dirname(log_file_path))
                self.log_file = open(log_file_path, 'w')
        self.test_with_train_split = test_with_train_split

        self.use_length_constraint = use_length_constraint
        self.length_regressor_coeff = length_regressor_coeff
        
        print("Use length constraint:", self.use_length_constraint)
        print("Length regressor coeff:", self.length_regressor_coeff)
        # input("BREAKPOINT")

        ds = load_dataset('GEM/schema_guided_dialog', split='test')
        self.act_id2name_map = {i : n for i, n in enumerate(ds.info.features['dialog_acts'][0]['act'].names)}


        self.dump_dag = dump_dag
        if dump_dag:
            self.dag_dump_dict = {'node_word_logits':[],  'node_word_idx':[], 'links':[], 'refs':[], 'forced_token_ids': [], 'valid_node_cnt': []}
        
        self.torch_precision = torch_precision
        print("torch_precision:", torch_precision)
        torch.set_float32_matmul_precision(torch_precision)
    
    def _init_model(self, model_config):
        da_model_config = DirectedAcyclicTransformerConfig(**model_config)
        self.model = DirectedAcyclicTransformer(da_model_config)
    
    def forward(self, x, *args, **kwargs):
        return self.model.nar_generate(x, *args, **kwargs, **self.test_config['generate_params'])
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        # x, mask = batch_depad(x, mask, pad_len=1)
        x, mask, y = batch_depad(x, mask, y, pad_len=1)
        intent_idx, slot_idx = batch.get('intent_idx'), batch.get('slot_idx')
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs['loss']
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        if outputs.get("glat_acc", None):
            self.log("glat_acc", outputs['glat_acc'], prog_bar=True, on_epoch=True, on_step=True, logger=True)
        if self.use_wandb:
            wandb.log({
                "train/loss": loss,
            })
            if outputs.get('glat_acc', None):
                wandb.log({
                    "train/glat_acc": outputs['glat_acc'],
                    "train/glat_context_p": outputs['glat_context_p'],
                    "train/glat_keep": outputs['glat_keep'],
                })
        return loss
    
    def on_validation_start(self) -> None:
        if self.local_rank != 0:
            return
        self.val_loss = []
        return super().on_validation_start()
    
    def validation_step(self, batch, batch_idx):
        if self.local_rank != 0:
            return
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        # x, mask = batch_depad(x, mask, pad_len=1)
        x, mask, y = batch_depad(x, mask, y, pad_len=1)
        # intent_idx, slot_idx = batch.get('intent_idx'), batch.get('slot_idx')
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs['loss']

        decoded_tokens_dict = self._decode_generative_step(x, y)
        log_dict = decoded_tokens_dict
        # log_dict['val_loss'] = loss.tolist()
        self.valid_eval_recorder.log_sample_dict_batch(log_dict)
        # for r, p, inp, out_degree_cnt, emit_degree_cnt, path, score, token_recall, inter_tokens in zip(decoded_tokens_dict['refs'], decoded_tokens_dict['preds'], decoded_tokens_dict['inps'], \
            # decoded_tokens_dict['out_degree_cnt'], decoded_tokens_dict['emit_degree_cnt'], decoded_tokens_dict['decoded_paths'], decoded_tokens_dict['scores'], decoded_tokens_dict['token_recall'], \
            # decoded_tokens_dict['intersected_tokens']):
            # self.valid_log_list.append((r, p, inp, out_degree_cnt, emit_degree_cnt, path, score, token_recall, inter_tokens))

        self.log("val_loss", loss, prog_bar=True)
        self.val_loss.append(loss)
        return loss
    
    def on_validation_end(self) -> None:
        if self.local_rank != 0:
            return
        valid_data = self.valid_eval_recorder.get_sample_logs()
        valid_df = pd.DataFrame(valid_data)
        eval_res = self._compute_eval_metrics(valid_df, self.whole_val_dataset)
        valid_df.to_csv(f"{self.trainer.log_dir}/valid_df-{self.valid_cnt}.csv")

        if self.use_wandb:
            valid_table = wandb.Table(data=eval_res['res_df'].values.tolist(), columns=eval_res['res_df'].columns.tolist())
            try:
                wandb.log({f"Validation Table-{self.valid_cnt}": valid_table})
            except Exception as err:
                print("Valid table error", err)
                print(valid_table)
                print(valid_df.head())
            dict_to_log = {
                f"val/Corpus_BLEU": eval_res['corpus_bleu'],
                f"val/loss": torch.tensor(valid_data['val_loss']).mean(),
                f"val/token_recall": eval_res['avg_token_recall'],
                f"val/Average Decoded Score": eval_res['avg_decoded_score'],
            }
            # wandb.log(dict_to_log, step=self.global_step, commit=True)
            wandb.log(dict_to_log)
            # wandb.log({f"val/Corpus_BLEU": eval_res['corpus_bleu']})
            # wandb.log({f"val/Corpus_SER": eval_res['corpus_ser']})
            # wandb.log({f"val/Applicable_SER": eval_res['applicable_ser']})
            # wandb.log({f"val/loss": torch.tensor(valid_data['val_loss']).mean()})
            # wandb.log({f"val/token_recall": eval_res['avg_token_recall']})
            # wandb.log({f"val/Average Decoded Score": eval_res['avg_decoded_score']})

        # if self.use_wandb:
            # wandb.log({f"val/{k}": v for k, v in valid_recorder_after_pipeilne.get_stats_logs().items()})
        return super().on_validation_end() 
    
    def setup(self, stage):
        data = self.dp.get_data([self.use_data_node], explode=True)
        use_columns = ['input_ids', 'labels', 'attention_mask']
        if stage in (None, "fit"):
            self.whole_val_dataset = copy.deepcopy(data['validation'])
            self.train_dataset = data['train']
            self.train_dataset.set_format('torch', columns=use_columns)
            self.val_dataset = data['validation']
            self.val_dataset.set_format('torch', columns=use_columns)
        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            self.whole_test_dataset = copy.deepcopy(data['test'])
            if self.test_with_train_split:
                self.test_dataset = data['train']
            else:
                self.test_dataset = data['test']
            self.test_dataset.set_format('torch', columns=use_columns) # set to None to support forced_ids
    
    def on_test_start(self):
        self.test_idx = 0
        self.test_start_time = time.time()
        return super().on_test_start()
        

    def test_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        x, mask, y = batch_depad(x, mask, y, pad_len=1)

        batch_size = batch['input_ids'].shape[0] 
        forced_token_ids = self.whole_test_dataset[self.test_idx:self.test_idx+batch_size]['forced_token_ids'] 
        self.test_idx += batch_size

        intent_idx, slot_idx = batch.get('intent_idx'), batch.get('slot_idx')

        specified_length_constraint = None
        if self.use_length_constraint == 'oracle':
            specified_length_constraint = torch.sum(y.ne(0), -1).tolist() # 0 is pad token
        elif self.use_length_constraint == 'regressor' and self.length_regressor_coeff is not None:
            intercept_, linear_term = float(self.length_regressor_coeff[0]), float(self.length_regressor_coeff[1])
            specified_length_constraint = [round(v) for v in intercept_ + torch.sum(x.ne(0),-1).cpu().numpy()*linear_term]
        else:
            specified_length_constraint = None
            # raise NotImplementedError(f'use_length_constraint = {self.use_length_constraint}: has not been implemented!')

        decoded_res_dict = self._decode_generative_step(x, y, forced_token_ids=forced_token_ids, specified_length_constraint=specified_length_constraint)
        print("="*100)
        print('ref:', decoded_res_dict['reference'])
        print('pred:', decoded_res_dict['prediction'])
        print("="*100)
        self.test_eval_recorder.log_sample_dict_batch(decoded_res_dict)
        # for r, p, inp, out_degree_cnt, emit_degree_cnt, path, score, token_recall, inter_tokens in zip(decoded_tokens_dict['refs'], decoded_tokens_dict['preds'], decoded_tokens_dict['inps'], \
            # decoded_tokens_dict['out_degree_cnt'], decoded_tokens_dict['emit_degree_cnt'], decoded_tokens_dict['decoded_paths'], decoded_tokens_dict['scores'], decoded_tokens_dict['token_recall'], \
            # decoded_tokens_dict['intersected_tokens']):
            # self.log_list.append((r, p, inp, out_degree_cnt, emit_degree_cnt, path, score, token_recall, inter_tokens))
        # if self.dump_dag:
            # self.dag_dump_dict['forced_token_ids'].extend(forced_token_ids)
        
        # outputs = self.forward(x)
        # refs = self.tokenizer.batch_decode(y, skip_special_toout_edgekens=True)
        # preds = self.tokenizer.batch_decode(outputs['output_tokens'], skip_special_tokens=True)
        # inps = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        # if self.save_decode:
        #     for r, p, inp in zip(refs, preds, inps):
        #         self.print(str(self.test_cnt)+'|', end='', file=self.log_file)
        #         self.print(r.replace('|','<vstripe>'), p.replace('|',"<vstripe>"), inp.replace('|','<vstripe>'), sep=' | ', file=self.log_file)
        #         self.test_cnt += 1
        #         self.log_list.append((r, p, inp))
    
    def on_test_end(self) -> None:
        test_end_time = time.time()
        self.test_eval_recorder.log_stats_dict({'test_time_seconds': test_end_time-self.test_start_time})
        if self.dump_dag:
            with open(self.log_file_path.parent / 'dumped_dag.json', 'w') as f:
                json.dump(self.dag_dump_dict, f)
            print(f"DAG saved to {self.log_file_path.parent} / 'dumped_dag.json")
        log_data = self.test_eval_recorder.get_sample_logs()
        test_df = pd.DataFrame(log_data)
        eval_res = self._compute_eval_metrics(test_df, self.whole_test_dataset)
        print("=======Evaluation results=======\n")
        disp_dict = {key: eval_res[key] for key in ['corpus_bleu', 'avg_token_recall', 'avg_decoded_score']}
        pprint(disp_dict)
            

        if self.use_wandb:
            test_table = wandb.Table(data=eval_res['res_df'].values.tolist(), columns=eval_res['res_df'].columns.tolist())
            wandb.log({f"Test Table": test_table})
            wandb.log({f"test/Corpus BLEU": eval_res['corpus_bleu']})
            wandb.log({f"test/Average Decoded Score": eval_res['avg_decoded_score']})
            wandb.log({f"test/loss": torch.tensor(self.val_loss).mean()})
        test_df.to_csv(self.log_file_path.parent / 'test_case.csv')
        
        with open(f"{self.log_file_path.parent}/evaluation_results.json", 'w') as f:
            json.dump(disp_dict, f)
        
        return super().on_test_end()
        
    def _prune_dag_to_dump(self, node_word_logits, links, forced_token_ids, top_k=3):
        """Note: only works for batch_size=1

        Args:
            node_word_logits (_type_): _description_
            links (_type_): _description_
            forced_token_ids (_type_): _description_
            top_k (int, optional): _description_. Defaults to 3.
        """
        all_forced_token_ids = []
        for t_ids in forced_token_ids[0]:
            all_forced_token_ids.extend(t_ids)

        _, top_idx = torch.topk(node_word_logits, top_k, dim=-1)

        batch_size, graph_size, _ = node_word_logits.shape

        select_idx = torch.zeros((batch_size, graph_size, top_k+len(all_forced_token_ids)), dtype=torch.long).to(node_word_logits.device)

        select_idx[:,:,:top_k] = top_idx
        select_idx[:,:,top_k:] = torch.tensor(all_forced_token_ids, dtype=torch.long).to(select_idx.device)

        pruned_node_word_logits = node_word_logits.gather(-1, select_idx)

        return pruned_node_word_logits, select_idx, links


    def _decode_generative_step(self, x, y=None, forced_token_ids=None, specified_length_constraint=None, inspect_dag=True):
        decode_stime = time.time()
        outputs = self.forward(x, forced_token_ids=forced_token_ids, specified_length_constraint=specified_length_constraint)
        decode_etime = time.time()
        batch_size = x.shape[0]
        avg_decode_time_in_batch = (decode_etime - decode_stime) / batch_size

        decoded_output = outputs['decoded_output']
        node_word_logits = outputs['node_word_logits']
        decoded_paths = decoded_output['decoded_paths']
        # print(decoded_output)
        # input("(BREAKPOINT)")
        links = outputs['links']
        refs = self.tokenizer.batch_decode(y, skip_special_tokens=True) if y is not None else None
        preds = None
        if 'output_strings' in decoded_output:
            preds = decoded_output['output_strings']
        else:
            preds = self.tokenizer.batch_decode(decoded_output['output_tokens'], skip_special_tokens=True)
        # preds = decoded_output.get('output_strings')
        inps = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        scores = decoded_output['output_scores']
        emit_degree_cnt = None
        token_recall = []

        decode_detail_infos = decoded_output.get('detail_infos', [{}]*len(refs))

       

        if inspect_dag:
            links_prob = links.exp()
            connection_mask = links_prob >= 0.2
            out_degree = links_prob.masked_fill(connection_mask, 1).masked_fill(~connection_mask, 0).sum(dim=-1).long()
            out_degree_cnt = self.batch_count_int_tensor(out_degree, 5)

            links_sum = links_prob.sum(dim=-1)
            valid_node_cnt = torch.zeros_like(links_sum, dtype=torch.long).masked_fill(links_sum>0.2, 1).sum(dim=-1)

            node_word_prob = F.softmax(node_word_logits, dim=-1)
            emission_mask = node_word_prob >= 0.2
            emit_degree = node_word_prob.masked_fill(emission_mask, 1).masked_fill(~emission_mask, 0).sum(dim=-1).long()
            for bs in range(batch_size):
                emit_degree[bs, valid_node_cnt[bs]:] = 0
            emit_degree_cnt = self.batch_count_int_tensor(emit_degree, 5)

            out_degree_cnt[:,0] = valid_node_cnt #  out_degree_cnt[batch_idx, 0] is the total number of valid nodes
            emit_degree_cnt[:, 0] = valid_node_cnt #  emit_degree_cnt[batch_idx, 0] is the total number of valid nodes

            max_emit_logits, emit_tokens = node_word_logits.max(dim=-1)
            intersects = self.torch_tensor_intersect(y, emit_tokens)
            for bs in range(batch_size):
                token_recall.append(len(intersects[bs])/len(torch.unique(y[bs]))) # len to resolve base case where tensor = [] (no intersection)
                
            if self.dump_dag:
                pruned_node_word_prob, select_idx, pruned_links = self._prune_dag_to_dump(node_word_prob, links, forced_token_ids)
                self.dag_dump_dict['node_word_logits'].append(pruned_node_word_prob[0].tolist())
                self.dag_dump_dict['node_word_idx'].append(select_idx[0].tolist())
                self.dag_dump_dict['links'].append(pruned_links[0].tolist())
                self.dag_dump_dict['valid_node_cnt'].append(valid_node_cnt.item())

        return {
            'reference': refs,
            'prediction': preds,
            'inps': inps,
            'decoded_score': scores,
            'out_degree_cnt': out_degree_cnt.tolist() if inspect_dag else [[-100] for _ in range(batch_size)],
            'emit_degree_cnt': emit_degree_cnt.tolist() if inspect_dag else [[-100] for _ in range(batch_size)],
            'decoded_paths': decoded_paths,
            'time_to_decode': [avg_decode_time_in_batch] * batch_size,
            'token_recall': token_recall,
            'intersected_tokens': intersects, 
            'decode_detail_infos': decode_detail_infos,
        }

    def batch_count_int_tensor(self, x, max_value):
        count_tgt = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
        count_tgt.scatter_add_(-1, x, torch.ones_like(x))
        return count_tgt
    
    def _compute_eval_metrics(self, df, whole_dataset_split):
        sentence_bleu = compute_sentence_bleu(df['prediction'].values.tolist(), df['reference'].values.tolist())
        corpus_bleu = compute_corpus_bleu(df['prediction'].values.tolist(), df['reference'].values.tolist())
        # sentence_ser, if_ser_applicable, corpus_ser, applicable_ser = compute_ser(whole_dataset_split, df['prediction'].values.tolist())
        df['bleu'] = sentence_bleu
        # df['slot_error'] = sentence_ser
        # df['ser_applicable'] = if_ser_applicable

       
        # df['full_dialog_acts'] = [[{'act': self.act_id2name_map[dact['act']], 'slot': dact['slot'], 'values': dact['values']} for dact in da] for da in whole_dataset_split['dialog_acts'][:len(df)]]
        # df['act_slot'] = ["|".join([f"{self.act_id2name_map[da['act']]}({da['slot']})" for da in dialog_acts]) for dialog_acts in whole_dataset_split['dialog_acts'][:len(df)]]
        avg_token_recall = df['token_recall'].mean()
        return {
            'res_df': df,
            'sentence_bleu': sentence_bleu,
            'corpus_bleu': corpus_bleu,
            'avg_token_recall': avg_token_recall,
            'avg_decoded_score': df['decoded_score'].mean(),
        }

    def torch_tensor_intersect(self, a, b):
        ac = a.detach().clone()
        bc = b.detach().clone()
        batch_size = a.shape[0]
        intersects = []
        for b_idx in range(batch_size):
            aa, bb = torch.unique(ac[b_idx]), torch.unique(bc[b_idx])
            aa, bb = self.pad_last_dim_to_same_shape(aa, bb)
            a_cat_b, counts = torch.cat([aa,bb]).unique(return_counts=True)
            intersected_tokens = a_cat_b[torch.where(counts.gt(1))] 
            intersects.append(intersected_tokens.tolist() if len(intersected_tokens) else [0])
        return intersects
    
    
    def pad_last_dim_to_same_shape(self, a, b):
        assert len(a.shape) == len(b.shape), f"to pad to same shape, no broadcasting is allowed: a.shape={a.shape}, b.shape={b.shape}"
        max_len = max(a.shape[-1], b.shape[-1])
        pad_shape = [0, 0] * len(a.shape)
        if a.shape[-1] < b.shape[-1]:
            pad_shape[1] = max_len - a.shape[-1]
            a = F.pad(a, pad_shape)
        else:
            pad_shape[1] = max_len - b.shape[-1]
            b = F.pad(b, pad_shape)
        return a, b
        
             