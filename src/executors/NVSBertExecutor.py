import torch
import pandas as pd
import os.path
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from data_ops.metrics import compute_vs_recall_with_vocab_prob
from models.modeling_nvs_bert import NVSBert
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import wandb

@register_executor
class NVSBertExecutor(BaseExecutor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        test_with_train_split=False,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, *args, **kwargs)
        self.use_data_node = use_data_node
        self.tokenizer = tokenizer
        if self.mode == 'test':
            self.save_decode = test_config.get('save_decode', True)
            if self.save_decode:
                if not os.path.exists(os.path.dirname(log_file_path)):
                    os.makedirs(os.path.dirname(log_file_path))
                self.log_file = open(log_file_path, 'w')
            self.pred_probs_arr = []
            self.labels_arr = []
        self.test_with_train_split = test_with_train_split

    
    def _init_model(self, model_config):
        self.model = NVSBert(**model_config)
    
    def forward(self, x, attention_mask, *args, **kwargs):
        outputs = self.model(input_ids=x, attention_mask=attention_mask)
        return outputs
        # return outputs.
        # return self.model.nar_generate(x, **self.test_config['generate_params'])
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['nvs_labels'], batch['attention_mask']
        x, mask = self.depad_x(x, attension_mask=mask)
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        if self.use_wandb:
            wandb.log({
                "train/loss": loss,
            })
        return loss
    
    def on_validation_start(self) -> None:
        self.valid_log_list = []
        self.valid_pred_probs_arr = []
        self.valid_labels_arr = []
        self.valid_cnt += 1
        return super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['nvs_labels'], batch['attention_mask']
        x, mask = self.depad_x(x, mask)
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)

        preds_vocab_idx = outputs['selected_vocab_idx']
        max_pool_vocab_sig = outputs['max_pool_vocab_sig']

        self.valid_pred_probs_arr.append(max_pool_vocab_sig)
        self.valid_labels_arr.append(y)

        preds_vocab_tokens = self.tokenizer.batch_decode(preds_vocab_idx, skip_special_tokens=True)
        gt_vocab_token_ids = self.get_token_id_from_binary_indicator(y)
        refs = self.tokenizer.batch_decode(gt_vocab_token_ids, skip_special_tokens=True)
        inps = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        for r, p, inp in zip(refs, preds_vocab_tokens, inps):
            self.valid_log_list.append((r, p, inp))

        if self.use_wandb:
            wandb.log({
                "val/loss": loss
            })
        return loss
    
    def setup(self, stage):
        data = self.dp.get_data([self.use_data_node], explode=True)
        use_columns = ['input_ids', 'nvs_labels', 'attention_mask']
        if stage in (None, "fit"):
            self.train_dataset = data['train']
            self.train_dataset.set_format('torch', columns=use_columns)
            self.val_dataset = data['validation']
            self.val_dataset.set_format('torch', columns=use_columns)
        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            if self.test_with_train_split:
                self.test_dataset = data['train']
            else:
                self.test_dataset = data['test']
            self.test_dataset.set_format('torch', columns=use_columns)
        
    def test_step(self, batch, batch_idx):
        # print(f'{__class__.__name__} is on cuda:', next(self.model.parameters()).is_cuda)
        x, y, mask = batch['input_ids'], batch['nvs_labels'], batch['attention_mask']
        x, mask = self.depad_x(x, mask)

        preds_res = self._predict_vocab(x, mask)
        preds_vocab_idx = preds_res['selected_vocab_idx']
        max_pool_vocab_sig = preds_res['max_pool_vocab_sig']

        preds_vocab_tokens = self.tokenizer.batch_decode(preds_vocab_idx, skip_special_tokens=True)
        gt_vocab_tokens = self.get_gt_token_id_from_binary()
        refs = self.tokenizer.batch_decode(gt_vocab_tokens, skip_special_tokens=True)
        inps = self.tokenizer.batch_decode(x, skip_special_tokens=True)

        self.pred_probs_arr.append(max_pool_vocab_sig)
        self.labels_arr.append(y)

        if self.save_decode:
            for r, p, inp in zip(refs, preds_vocab_tokens, inps):
                # self.print(str(self.test_cnt)+'|', end='', file=self.log_file)
                # self.print(r.replace('|','<vstripe>'), p.replace('|',"<vstripe>"), inp.replace('|','<vstripe>'), sep=' | ', file=self.log_file)
                self.test_cnt += 1
                self.log_list.append((r, p, inp))
    
    def on_test_end(self) -> None:
        self.labels_mat = torch.cat(self.labels_arr)
        self.pred_probs_mat = torch.cat(self.pred_probs_arr)

        eval_input_data = {
            'preds': self.pred_probs_mat,
            'labels': self.labels_mat
        }

        input_data_dict = {input_node_id: eval_input_data for input_node_id in self.eval_pipeline.input_transform_ids}
        eval_res = self.eval_pipeline.apply_transforms(input_data_dict=input_data_dict)
        
        test_df = pd.DataFrame(self.log_list, columns=['reference', 'prediction', 'input'])
        for top_k in eval_res:
            test_df[f"k={top_k}-recall"] = eval_res['example_recall']

        test_df.to_csv(self.log_file_path.parent / 'test_case.csv')
    
    def on_validation_end(self):
        self.valid_labels_mat = torch.cat(self.valid_labels_arr)
        self.valid_pred_probs_mat = torch.cat(self.valid_pred_probs_arr)

        # eval_input_data = {
        #     'preds': self.valid_pred_probs_mat,
        #     'labels': self.valid_labels_mat
        # }

        # input_data_dict = {input_node_id: eval_input_data for input_node_id in self.eval_pipeline.input_transform_ids}
        # eval_res = self.eval_pipeline.apply_transforms(input_data_dict=input_data_dict)['compute_recall']
        eval_res = compute_vs_recall_with_vocab_prob(self.valid_pred_probs_mat, self.valid_labels_mat, top_k_list=[50, 100, 200, 500])
        
        valid_df = pd.DataFrame(self.valid_log_list, columns=['reference', 'prediction', 'input'])
        print("Validation:", self.valid_cnt)
        for top_k in eval_res:
            valid_df[f"k={top_k}-recall"] = eval_res[top_k]['example_recall']
            print(f"k={top_k}-recall", eval_res[top_k]['avg_recall'])

        if self.use_wandb:
            valid_table = wandb.Table(data=valid_df.values.tolist(), columns=valid_df.columns.tolist())
            wandb.log({f"Validation Table-{self.valid_cnt}": valid_table})
        print("="*8)
        

        # valid_df.to_csv(self.train_dir / f"ValidationTable-{self.valid_cnt}.csv") 


    def _predict_vocab(self, x, mask):
        pred_res = self.forward(x, attention_mask=mask)
        return pred_res
    
    def get_token_id_from_binary_indicator(self, y):
        tgt_lengths = y.sum(dim=-1)
        token_idx = y.nonzero(as_tuple=True)[1]
        padded_token_idx = torch.nn.utils.rnn.pad_sequence(token_idx.split(tgt_lengths.tolist())).T
        return padded_token_idx
    
    def depad_x(self, x, attension_mask=None):
        max_in_length = torch.max((~(x==0)).sum(dim=-1))
        attension_mask = attension_mask[:, :max_in_length]
        return x[:, :max_in_length], attension_mask
        

