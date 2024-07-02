import pandas as pd
import os.path
from runway_for_ml.executors.base_executor import BaseExecutor
from runway_for_ml.utils.global_variables import register_executor
from runway_for_ml.utils.util import batch_depad
from torch.utils.data import DataLoader
from runway_for_ml.configs.configuration import (
    DataPipelineConfig,
    ModelConfig,
)
import wandb
import time
import copy

@register_executor
class T5Executor(BaseExecutor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        use_constrained_decoding=False,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, use_data_node=use_data_node, *args, **kwargs)
        self.tokenizer = tokenizer
        self.use_constraint_decoding = use_constrained_decoding
        # if self.mode == 'test':
        #     self.save_decode = test_config.get('save_decode', True)
        #     if self.save_decode:
        #         if not os.path.exists(os.path.dirname(log_file_path)):
        #             os.makedirs(os.path.dirname(log_file_path))
        #         self.log_file = open(log_file_path, 'w')

    def forward(self, x, *args, forced_token_ids=None, **kwargs):
        if forced_token_ids is not None and len(forced_token_ids[0][0]) and self.use_constraint_decoding: #NOTE: only support batch_size=1
            return self.model.generate(x, force_words_ids=forced_token_ids[0], **self.test_config['generate_params'])
        else:
            return self.model.generate(x, **self.test_config['generate_params'])
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        # x, mask, y = batch_depad(x, mask, y, pad_len=1)
        intent_idx, slot_idx = batch.get('intent_idx'), batch.get('slot_idx')
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss

        self.log("train/train_loss", loss, prog_bar=True, on_step=True, logger=True, sync_dist=True)
        if self.use_wandb:
            wandb.log({
                "train/loss": loss,
            })
        return loss

    def on_validation_start(self) -> None:
        return super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        # x, mask, y = batch_depad(x, mask, y, pad_len=1)
        # intent_idx, slot_idx = batch.get('intent_idx'), batch.get('slot_idx')
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs.loss

        decoded_tokens_dict = self._decode_generative_step(x, y)
        log_dict = decoded_tokens_dict
        # print(len(loss))
        self.valid_eval_recorder.log_sample_dict_batch(decoded_tokens_dict)
            # for r, p, inp in zip(decoded_tokens_dict['refs'], decoded_tokens_dict['preds'], decoded_tokens_dict['inps']):
                # self.valid_log_list.append((r, p, inp))

        self.log("val_loss", loss, prog_bar=True)
        if self.use_wandb:
            wandb.log({
                "val/loss": loss
            })
        return {
            "loss": loss
        }

    def on_validation_end(self) -> None:
        valid_log_data = self.valid_eval_recorder.get_sample_logs()
        print(valid_log_data)
        # valid_df = pd.DataFrame(self.valid_log_list, columns=['reference', 'prediction', 'input'])
        # if self.use_wandb:
        #     valid_table = wandb.Table(data=valid_df.values.tolist(), columns=valid_df.columns.tolist())
        #     wandb.log({f"Validation Table-{self.valid_cnt}": valid_table})
        return super().on_validation_end()
    
    def setup(self, stage):
        super().setup(stage)
        data = self.dp.get_data([self.use_data_node], explode=True)
        use_columns = ['input_ids', 'labels', 'attention_mask']
        if stage in (None, "fit"):
            self.train_dataset = data['train']
            self.train_dataset.set_format('torch', columns=use_columns)
            self.val_dataset = data['validation']
            self.val_dataset.set_format('torch', columns=use_columns)
        # Assign Test split(s) for use in Dataloaders
        if stage in (None, "test"):
            self.whole_test_dataset = copy.deepcopy(data['test'])
            self.test_dataset = data['test']
            self.test_dataset.set_format('torch', columns=use_columns)
            if "DART" in self.whole_test_dataset.info.description:
                self.test_triplesets = self.whole_test_dataset['tripleset']


    def on_test_start(self):
        self.test_idx = 0
        self.test_start_time = time.time()
        return super().on_test_start()

        

    def test_step(self, batch, batch_idx):
        # print(f'{__class__.__name__} is on cuda:', next(self.model.parameters()).is_cuda)
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']

        batch_size = x.shape[0]
        forced_token_ids = self.whole_test_dataset[self.test_idx:self.test_idx+batch_size]['forced_token_ids'] 
        self.test_idx += batch_size
        
        # x, mask, y = batch_depad(x, mask, y, pad_len=1)
        intent_idx, slot_idx = batch.get('intent_idx'), batch.get('slot_idx')
        decoded_tokens_dict = self._decode_generative_step(x, y, forced_token_ids=forced_token_ids)
        # for r, p, inp in zip(decoded_tokens_dict['refs'], decoded_tokens_dict['preds'], decoded_tokens_dict['inps']):
            # self.log_list.append((r, p, inp))
        log_dict = decoded_tokens_dict
        if "DART" in self.whole_test_dataset.info.description:
            triples = [t for t in self.test_triplesets[batch_idx*batch_size:(batch_idx+1)*batch_size]]
            log_dict.update({'triples': triples})
        self.test_eval_recorder.log_sample_dict_batch(log_dict)

        # outputs = self.forward(x)
        # refs = self.tokenizer.batch_decode(y, skip_special_tokens=True)
        # preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
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
        super().on_test_end()
        # test_df = pd.DataFrame(self.log_list, columns=['reference', 'prediction', 'input'])
        # test_df.to_csv(self.log_file_path / 'test_case.csv')
    
    def _decode_generative_step(self, x, y=None, forced_token_ids=None):
        outputs = self.forward(x, forced_token_ids=forced_token_ids)
        refs = self.tokenizer.batch_decode(y, skip_special_tokens=True) if y is not None else None
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        inps = self.tokenizer.batch_decode(x, skip_special_tokens=True)
        # breakpoint()
        return {
            'reference': refs,
            'prediction': preds,
            'inps': inps
        }