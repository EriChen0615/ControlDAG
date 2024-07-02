import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers import BertModel
from typing import List, Optional, Tuple, Union
from easydict import EasyDict

class NVSBert(nn.Module):
    def __init__(
        self,
        bert_model_version,
        top_k, # top k for vocabulary selection
        lambda_p, # weight for positive instances 
        lambda_thresh=None, # threshold lambda for vocabulary selection, NOT USED
    ):
        super().__init__()
        self.bert_model_version = bert_model_version
        self.bert_model = BertModel.from_pretrained(self.bert_model_version)
        self.bert_config = self.bert_model.config
        bert_device = self.bert_model.device

        self.lambda_p = lambda_p
        self.lambda_thresh = lambda_thresh
        self.top_k = top_k
        self.vocab_size = self.bert_config.vocab_size
        self.bert_hidden_size = self.bert_config.hidden_size
        self.vs_head = nn.Linear(self.bert_hidden_size, self.vocab_size, device=bert_device)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        bert_out = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        bert_last_hidden_state = bert_out.last_hidden_state
        vocab_projection = self.vs_head(bert_last_hidden_state)

        max_pool_vocab_logits, max_pool_vocab = torch.max(vocab_projection, dim=1)
        max_pool_vocab_sig = torch.sigmoid(max_pool_vocab_logits)

        # selected_vocab_idx = torch.nonzero(max_pool_vocab_logits >= self.lambda_thresh)
        selected_vocab_prob, selected_vocab_idx = torch.topk(max_pool_vocab_sig, self.top_k, dim=1)

        loss = None
        if labels is not None:
            weights = torch.ones_like(labels, dtype=torch.float)
            weights[labels==1] = self.lambda_p
            loss = F.binary_cross_entropy(max_pool_vocab_sig, labels.float(), weight=weights)

        
        return EasyDict({
            'loss': loss,
            'selected_vocab_prob': selected_vocab_prob,
            'selected_vocab_idx': selected_vocab_idx,
            'max_pool_vocab_sig': max_pool_vocab_sig,
        })
        





