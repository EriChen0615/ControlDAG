import torch
import torch.nn.functional as F
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

@register_executor
class BertMRPCExecutor(BaseExecutor):
    def __init__(self,
        data_pipeline_config: DataPipelineConfig,
        model_config: ModelConfig,
        mode, # train/infer/eval
        train_config={},
        test_config={},
        log_file_path=None,
        use_data_node=None,
        tokenizer=None,
        global_config=None,
        *args, **kwargs
        ):
        super().__init__(data_pipeline_config, model_config, mode, train_config=train_config, test_config=test_config, log_file_path=log_file_path, input_data_node=input_data_node, global_config=global_config, *args, **kwargs)
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    
    def _init_model(self, model_config): 
        """Initialize self.model

        Args:
            model_config (dict): contains key-values for model configuration
        """
        # super()._init_model(model_config) # alternatively, use the default implementation in super()._init_model()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
        )
    
    def prepare_data(self):
        super().prepare_data()
    
    def setup(self, stage):
        use_columns = ['input_ids', 'token_type_ids', 'attention_mask', 'labels']
        if stage in (None, "fit"):
            self.train_dataset = self.prepared_data['train']
            self.train_dataset.set_format('torch', columns=use_columns)
            self.val_dataset = self.prepared_data['validation']
            self.val_dataset.set_format('torch', columns=use_columns)
        else:
            self.test_dataset = self.prepared_data['test']
            self.test_dataset.set_format('torch', columns=use_columns)
            
    def training_step(self, batch, batch_idx):
        """Defines training step for each batch

        Args:
            batch (_type_): _description_
            batch_idx (_type_): _description_
        """
        x, y, mask = batch['input_ids'], batch['labels'], batch['attention_mask']
        outputs = self.model(input_ids=x, labels=y, attention_mask=mask)
        loss = outputs[-1]
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True, logger=True)
        return loss

