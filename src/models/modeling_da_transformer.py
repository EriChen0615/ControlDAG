from torch.nn import Transformer
from transformers.modeling_utils import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Stack 
from transformers.models.t5.configuration_t5 import T5Config
from .configuration_da_transformer import DirectedAcyclicTransformerConfig
from .modeling_da_utils import (
    PositionalEmbedding
)

from .nar_generation_utils import NARGenerationMixin
from .modeling_nar_t5 import NAR_T5ForConditionalGeneration, NAR_T5Stack


import torch
from torch import nn
import torch.nn.functional as F
# import modeling_utils
# from modeling_utils import logsumexp
from typing import Optional, Tuple, Union

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput
)

from transformers.models.t5.modeling_t5 import (
    T5LayerFF,
    T5LayerNorm,
    T5LayerCrossAttention,
    T5LayerSelfAttention,
)

from transformers.models.t5.configuration_t5 import (
    T5Config
)

from easydict import EasyDict

DUMMY_INPUTS = [[7, 6, 0, 0, 0], [1, 2, 3, 0, 0], [4, 5, 0, 0, 0]]
DUMMY_MASK = [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]
DUMMY_LABELS = [[7, 6, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 3, 4, 5]]

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    # max_trg_len
    index_t = new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


def logsumexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    m, _ = x.max(dim=dim)
    mask = m == -float('inf')
    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float('inf'))
 
def torch_dag_logsoftmax_gather_inplace(node_word_logits, select_idx):
    r""" Fused operation of log_softmax and gather"""
    r"""Comments:
    logits.shape [104, 312, 42728]
    word_ins_out.shape = [104, 312, 43728]
    select_idx.shape = [104, 312, 30] = tgt_tokens.unsqueeze(1).expand(-1, prelen, -1)
    """
    logits = torch.log_softmax(node_word_logits, -1, dtype=torch.float32)
    node_word_log_prob_on_tgt_tokens = logits.gather(dim=-1, index=select_idx)
    return node_word_log_prob_on_tgt_tokens

def restore_valid_links(links):
    # batch * prelen * trans_len
    batch_size, prelen, translen = links.shape
    valid_links_idx = torch.arange(prelen, dtype=torch.long, device=links.device).unsqueeze(1) + \
                torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0) + 1
    invalid_idx_mask = valid_links_idx >= prelen
    valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
    res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
    res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
    return res[:, :, :prelen]

def logsumexp_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float('inf')
    m = m.detach()
    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    return s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, -float('inf'))

def loop_function_noempty(last_f: torch.Tensor, links: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
    f_next = logsumexp_keepdim(last_f + links, 1) # batch * 1 * prelen
    f_next = f_next.transpose(1, 2) + match # batch * prelen * 1
    return f_next

def loop_function_noempty_max(last_f: torch.Tensor, links: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
    f_next = torch.max(last_f + links, dim=1)[0] # batch * 1 * prelen
    f_next = f_next.unsqueeze(-1) + match # batch * prelen * 1
    return f_next

def __torch_max_loss(match_all, links, output_length, target_length):
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"

    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    r"""
    f_init.shape = [104, 312, 1]
    match_all.shape = [104, 312, 30]
    len(match_arr) = 30
    The code below **accumulates** the probabilities of target tokens
    """

    match_arr = torch.chunk(match_all, tarlen, -1)
    for i in range(1, tarlen):
        f_now = loop_function_noempty_max(f_arr[-1], links, match_arr[i])
        f_arr.append(f_now)

    alllogprob = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return alllogprob

def torch_dag_best_alignment(match_all, links, output_length, target_length):
    r"""
    Function to obtain the alignment between prediction and reference
    Input:
        match_all (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_target_length, max_output_length]
            match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
            based on the j-th vertex.
            (Note: float32 are preferred; float16 may cause precision problem)
        links (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_output_length, max_transition_length]
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
        output_length (torch.LongTensor):
            Shape: [batch_size]
            output_length should be the graph size, the vertices (index >= graph size) are ignored
        target_length (torch.LongTensor):
            Shape: [batch_size]
            target_length is the reference length, the tokens (index >= target length) are ignored

    Output (torch.LongTensor):
        Shape: [batch_size, max_output_length]
        if output[b, i]>=0, it represents the index of target token aligned with the i-th vertex
        otherwise, output[b, i] = -1, it represents the i-th vertex is not aligned with any target token
    """
    with torch.enable_grad():
        match_all.requires_grad_()
        alllogprob = __torch_max_loss(match_all, links, output_length, target_length)
        matchgrad = torch.autograd.grad(alllogprob.sum(), [match_all])[0] # batch * talen * prelen
    pathvalue, path = matchgrad.max(dim=1)
    path.masked_fill_(pathvalue < 0.5, -1)
    return path

def torch_dag_loss(match_all, links, output_length, target_length):
    r"""
    Function to calculate the dag loss.
    Input:
        match_all (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_target_length, max_output_length]
            match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
            based on the j-th vertex.
            (Note: float32 are preferred; float16 may cause precision problem)
        links (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_output_length, max_transition_length]
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
        output_length (torch.LongTensor):
            Shape: [batch_size]
            output_length should be the graph size, the vertices (index >= graph size) are ignored
        target_length (torch.LongTensor):
            Shape: [batch_size]
            target_length is the reference length, the tokens (index >= target length) are ignored

    Output (torch.FloatTensor or torch.HalfTensor):
        Shape: [batch_size]
        the loss of each sample
    """
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"

    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    match_all_chunk = torch.chunk(match_all, tarlen, -1) # k * [batch * prelen * 1]

    for k in range(1, tarlen):
        f_now = loop_function_noempty(f_arr[-1], links, match_all_chunk[k])
        f_arr.append(f_now)

    loss_result = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return loss_result


class DirectedAcyclicDecoder(nn.Module):
    def __init__(
        self,
        base_decoder, # A T5 decoder by default
        lm_head, # A T5 Language model head by default
        link_features="decoder_out:position",
        positional_embed="learned",
    ):
        super().__init__()
        self.base_decoder = base_decoder
        self.lm_head = lm_head
        self.base_decoder_config = base_decoder.config
        self.decoder_dim = self.base_decoder_config.d_model
        self.padding_idx = self.base_decoder_config.pad_token_id
        # self.dropout_module = nn.Dropout(self.base_decoder_config.dropout_rate)

        if positional_embed == "learned":
            self.embed_positions = PositionalEmbedding( 
                num_embeddings=1024, # hard coded maxium target length
                embedding_dim=self.base_decoder_config.d_model,
                padding_idx=self.padding_idx,
            )
        else:
            self.embed_positions = None
        
        self.dropout_module = torch.nn.Dropout(0.1)

        self.embed_tokens = self.base_decoder.embed_tokens # Correct, share the input embeddings of the decoder.
        self.layer_norm = T5LayerNorm(
            self.decoder_dim,
            eps=self.base_decoder_config.layer_norm_epsilon
        )
        self.in_projection_layer = None
        self.out_projection_layer = nn.Linear( # output projection layer
            in_features=self.decoder_dim, 
            out_features=self.decoder_dim
        )

        self._init_link_structure(link_features)

        
    
    def _init_link_structure(self, link_features):
        self.link_features = link_features.split(":")
        links_dim = 0
        if "decoder_out" in self.link_features:  # use decoder out features
            links_dim += self.decoder_dim
        if "position" in self.link_features:
            self.link_positional = PositionalEmbedding(
                1024,
                self.decoder_dim,
                self.base_decoder_config.pad_token_id,
            ) #TODO handle pad properly
            links_dim += self.decoder_dim
        elif "sinposition" in self.link_features:
            links_dim += self.decoder_dim
        else:
            self.link_positional = None
        
        self.links_dim = links_dim

        self.link_query_linear = nn.Linear(
            self.links_dim,
            self.decoder_dim,
        )
        self.link_key_linear = nn.Linear(
            self.links_dim,
            self.decoder_dim,
        )
        self.link_gate_linear = nn.Linear(
            self.links_dim,
            self.base_decoder_config.num_heads,
        )

    def forward(
        self,
        normalize,
        encoder_out,
        encoder_out_mask,
        decoder_input_ids,
        step=0,
        **kwargs
    ):
        features, _ = self.extract_word_logits_and_links(
            decoder_input_ids,
            encoder_out=encoder_out,
            encoder_out_mask=encoder_out_mask,
            embedding_copy=(step == 0) & self.src_embedding_copy,
        )
        decoder_out = self.output_layer(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out
    
    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding

    def forward_embedding(self, decoder_input_ids, states=None):
        # embed positions
        positions = (
            self.embed_positions.forward_positional(decoder_input_ids)
            if self.embed_positions is not None
            else None
        )

        # embed tokens and positions
        if states is None:
            x = self.embed_tokens(decoder_input_ids)
            if self.in_projection_layer is not None:
                x = self.in_projection_layer(x)
        else:
            x = states

        if positions is not None:
            x += positions
        x = self.dropout_module(x)
        decoder_padding_mask = decoder_input_ids.eq(self.padding_idx)
        return x, decoder_padding_mask

    def extract_word_logits_and_links(
        self,
        decoder_input_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        early_exit=None,
        **kwargs
    ):
        """
        Similar to *forward* but features and links.

        Inputs:
            prev_output_tokens: Tensor(B, T)
            encoder_out: Tuple of (
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            )
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
            the LevenshteinTransformer decoder has full-attention to all generated tokens
        """
        # embedding
        x, decoder_padding_mask = self.forward_embedding(decoder_input_ids) 

        # B x T x C -> T x B x C #TODO Why do we need this?
        # x = x.transpose(0, 1)
        r"""
        x is positional embeddings + decoder input tokens embeddings (with GLAT)
        STEP 3a. Obtain decoder input embeddings from decoder input_ids
        STEP 3b. Obtain the vectors that corresponds to each node (x)
        STEP 3c. Obtain the output token distribution (node_word_logits) that corresponds to each node
        """

        base_decoder_output = self.base_decoder(
            inputs_embeds=x,
            attention_mask=(decoder_padding_mask==False), #NOTE: invert decoder_padding_mask because attention_mask=0 is masked
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False, # must set to False, otherwise is_decoder must be true 
        )

        x = base_decoder_output[0]

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C #TODO Why do we need this?
        # x = x.transpose(0, 1)

        if self.out_projection_layer is not None:
            x = self.out_projection_layer(x)
        
        node_word_logits = self.lm_head(x) # lm head is just a simple linear projection
        # node_word_logits = F.log_softmax(node_word_logits, dim=-1) # get normalized logits

        r"""
        STEP 4. Obtain the links between the nodes (i.e., transition probabilities of the DAG)
        """
        links = self.extract_links(x, decoder_input_ids)

        return node_word_logits, links

    def extract_links(
        self, 
        features, 
        prev_output_tokens,
    ):
        r"""
        features: decoder output (cross-attn)
        prev_output_tokens: [<bos> <pad> ... <pad> <eos>]
        self.link_positional: the positional embeddings for the links. N.B. This is not the same as positional embeddings at decoder input 
        """
        links_feature_arr = []
        if "decoder_out" in self.link_features:
            links_feature_arr.append(features)
        if "position" in self.link_features or "sinposition" in self.link_features:
            links_feature_arr.append(self.link_positional.forward_positional(prev_output_tokens))

        features_withpos = torch.cat(links_feature_arr, dim=-1)
        r"""
        features_withpos.shape = [104, 312, 1024]
        """

        batch_size = features.shape[0]
        seqlen = features.shape[1]
        chunk_num = self.base_decoder_config.num_heads
        chunk_size = self.decoder_dim // self.base_decoder_config.num_heads
        ninf = float("-inf")
        target_dtype = torch.float

        r"""
        Self-attention of concat(decoder_out, self.link_positional_embeds)
        """
        query_chunks = self.link_query_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        key_chunks = self.link_key_linear(features_withpos).reshape(batch_size, seqlen, chunk_num, chunk_size)
        r"""
        query_chunks.shape = [104, 312, 8, 64]
        key_chunk.shape = [104, 312, 8, 64]
        """
        log_gates = F.log_softmax(self.link_gate_linear(features_withpos), dim=-1, dtype=target_dtype) # batch_size * seqlen * chunk_num
        log_multi_content = (torch.einsum("bicf,bjcf->bijc", query_chunks.to(dtype=target_dtype), key_chunks.to(dtype=target_dtype)) / (chunk_size ** 0.5))
        r"""
        self.gate_linear = Linear(in_feature=1024, out_features=8, bias=True)
        log_gates.shape = [104, 312, 8]
        log_multi_content.shape = [104, 312, 312, 8]
        links.shape = [104, 312, 311] transition probabilities except itself.
        """
        # previous link computation
        # log_multi_content = F.log_softmax(log_multi_content, dim=2)
        # links = logsumexp(log_multi_content + log_gates.unsqueeze(2), dim=-1) 
        # links.triu_(1) # make links upper-triangular, self-transition disallowed
        # links[links==0] = ninf # log-likelihood
        
        # Original implementation
        log_multi_content_extract, link_nouse_mask = self.extract_valid_links(log_multi_content, prev_output_tokens.ne(self.padding_idx))
                    # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
        log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
        log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
        log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
        links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        links = restore_valid_links(links) # make it square

        # original code, where the link shape is one-off TODO: understand difference
        # if self.args.max_transition_length != -1:
        # log_multi_content_extract, link_nouse_mask = self.extract_valid_links(log_multi_content, prev_output_tokens.ne(self.padding_idx))
                # batch * seqlen * trans_len * chunk_num, batch * seqlen * trans_len
        # log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
        # log_multi_content_extract = F.log_softmax(log_multi_content_extract, dim=2)
        # log_multi_content_extract = log_multi_content_extract.masked_fill(link_nouse_mask.unsqueeze(-1).unsqueeze(-1), ninf)
        # links = logsumexp(log_multi_content_extract + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * trans_len
        # else:
        # link_mask = torch.ones(seqlen, seqlen, device=prev_output_tokens.device, dtype=bool).triu_(1).unsqueeze(0) & prev_output_tokens.ne(self.padding_idx).unsqueeze(1)
        # link_nouse_mask = link_mask.sum(dim=2, keepdim=True) == 0
        # link_mask.masked_fill_(link_nouse_mask, True)
        # log_multi_content.masked_fill_(~link_mask.unsqueeze(-1), ninf)
        # log_multi_attention = F.log_softmax(log_multi_content, dim=2)
        # log_multi_attention = log_multi_attention.masked_fill(link_nouse_mask.unsqueeze(-1), ninf)
        # links = logsumexp(log_multi_attention + log_gates.unsqueeze(2), dim=-1) # batch_size * seqlen * seqlen

        return links
    
    def extract_valid_links(self, content, valid_mask):
        # batch * prelen * prelen * chunk, batch * prelen
        r"""
        content.shape = [104, 312, 312, 8] (B, T, T, AttnChunk)
        valid_mask.shape = [104, 312]
        valid_links_idx = [104, 312, 311]
        """

        prelen = content.shape[1]
        translen = 1024 # self.args.max_transition_length
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=content.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=content.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= valid_mask.sum(dim=-1, keepdim=True).unsqueeze(-1)
        valid_links_idx = valid_links_idx.unsqueeze(0).masked_fill(invalid_idx_mask, 0) # Upper Ttiangular Mask

        res = content.gather(2, valid_links_idx.unsqueeze(-1).expand(-1, -1, -1, content.shape[-1]))
        res.masked_fill_(invalid_idx_mask.unsqueeze(-1), float("-inf"))

        return res, invalid_idx_mask.all(-1) # batch * prelen * trans_len * chunk, batch * prelen * trans_len
    
   

class DirectedAcyclicTransformer(PreTrainedModel, NARGenerationMixin):
    def __init__(
        self,
        config: DirectedAcyclicTransformerConfig,
        **kwargs
    ):
        super().__init__(config)
        self.config = config
        self.base_model_class = config.base_model_class
        self.base_model_version = config.base_model_version
        self.gen_decoder_input_args = EasyDict(config.gen_decoder_input_args)
        self.use_glat = config.use_glat
        self.glat_params = config.glat_params
        self.use_pretrained_base = config.use_pretrained_base

        if self.base_model_class == 'T5':
            self._apply_t5_init(self.base_model_version, self.use_pretrained_base, self.config.use_pretrained_encoder)
        elif self.base_model_class == 'NAR_T5':
            self._apply_nar_t5_init(self.base_model_version, self.use_pretrained_base, self.config.use_pretrained_encoder)
        
        self.nat_decoder = DirectedAcyclicDecoder(
            self.base_decoder,
            self.lm_head,
        )

    def dummy_input(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        labels = torch.tensor(DUMMY_LABELS)
        dummy_inputs = {
            "input_ids": input_ids,
            "labels": labels,
        }
        return dummy_inputs
    

    def _set_base_decoder_to_encoder_mode(self):
        """
        Turn masked attention to self attention
        """
        self.base_decoder.is_decoder = False
        self.base_decoder.config.is_decoder = False
        for block in self.base_decoder.block:
            block.is_decoder = False
            for layer in block.layer:
                layer.is_decoder = False
                if isinstance(layer, T5LayerCrossAttention):
                    layer.EncDecAttention.is_decoder = False
                elif isinstance(layer, T5LayerSelfAttention):
                    layer.SelfAttention.is_decoder = False
                elif isinstance(layer, T5LayerFF) or isinstance(layer, T5LayerNorm):
                    pass
                else:
                    raise NotImplementedError("Some layer is not set to is_decoder=False")
    
    def _apply_nar_t5_init(self, model_version, use_pretrained_base=False, use_pretrained_encoder=False):
        t5_model = None
        if use_pretrained_base or use_pretrained_encoder:
            t5_model = NAR_T5ForConditionalGeneration.from_pretrained(model_version)
        else:
            t5_config = T5Config.from_pretrained(model_version) 
            t5_model = NAR_T5ForConditionalGeneration(t5_config)

        self.base_config = t5_model.config
        self.base_encoder = t5_model.get_encoder()
        self.base_decoder = t5_model.get_decoder()

        if use_pretrained_encoder and not use_pretrained_base:
            for decoder_layer in self.base_decoder.block:
                decoder_layer.apply(t5_model._init_weights) # reset weights
            
        if not use_pretrained_base or self.config.reinit_lm_head:
            t5_model.lm_head.apply(t5_model._init_weights)
            # self.base_decoder = NAR_T5Stack.from_pretrained(model_version, embed_tokens=self.base_encoder.get_input_embeddings())
            # self.base_decoder.config.is_decoder = True
            # self.base_decoder.config.is_encoder_decoder = True
        
        # if self.config.reinit_lm_head and use_pretrained_base: # if use pretrained base, default to NOT reinit language model head
        #     self.lm_head = nn.Linear(self.base_config.d_model, self.base_config.vocab_size, bias=False)
        # else:
        self.lm_head = t5_model.lm_head
        
        # assigned to use code from da-transformer
        self.pad = self.base_config.pad_token_id
        self.eos = self.base_config.eos_token_id
        
        if 'unk_token_id' not in vars(self.base_config) or self.base_config.unk_token_id is None or 'bos_token_id' not in vars(self.base_config) or self.base_config.bos_token_id is None: 
            self._add_unk_bos_to_base_decoder()

        self.unk = self.base_config.unk_token_id
        self.bos = self.base_config.bos_token_id

    def _apply_t5_init(self, model_version, use_pretrained_base=False, use_pretrained_encoder=True):
        t5_model = None
        if use_pretrained_base or use_pretrained_encoder:
            t5_model = T5ForConditionalGeneration.from_pretrained(model_version)
        else:
            t5_config = T5Config.from_pretrained(model_version) 
            t5_model = T5ForConditionalGeneration(t5_config)

        self.base_config = t5_model.config
        self.base_encoder = t5_model.get_encoder()

        if use_pretrained_encoder and not use_pretrained_base:
            self.base_decoder = T5Stack.from_pretrained(model_version, embed_tokens=self.base_encoder.get_input_embeddings())
            self.base_decoder.config.is_decoder = True
            self.base_decoder.config.is_encoder_decoder = True
        else:
            self.base_decoder = t5_model.get_decoder()
        
        if self.config.reinit_lm_head and use_pretrained_base: # if use pretrained base, default to NOT reinit language model head
            self.lm_head = nn.Linear(self.base_config.d_model, self.base_config.vocab_size, bias=False)
        else:
            self.lm_head = t5_model.lm_head
        # self._set_base_decoder_to_encoder_mode() # turn masked attention to self attention
        
        # assigned to use code from da-transformer
        self.pad = self.base_config.pad_token_id
        self.eos = self.base_config.eos_token_id
        
        if 'unk_token_id' not in vars(self.base_config) or self.base_config.unk_token_id is None or 'bos_token_id' not in vars(self.base_config) or self.base_config.bos_token_id is None: 
            self._add_unk_bos_to_base_decoder()

        self.unk = self.base_config.unk_token_id
        self.bos = self.base_config.bos_token_id

    def _add_unk_bos_to_base_decoder(self):
        orig_num_embeddings = self.base_decoder.get_input_embeddings().num_embeddings
        self.base_decoder.resize_token_embeddings(orig_num_embeddings+2)
        bos_token_idx = orig_num_embeddings
        unk_token_idx = orig_num_embeddings + 1
        self.base_config.bos_token_id = bos_token_idx
        self.base_config.unk_token_id = unk_token_idx

    ### METHODS FROM T5 START ###
    def get_encoder(self):
        return self.base_encoder

    def get_decoder(self):
        return self.base_decoder
    ### METHODS FROM T5 END ###

    def restore_valid_links(self, links):
        # batch * prelen * trans_len
        batch_size, prelen, translen = links.shape
        translen: int = 1024
        if translen > prelen - 1:
            translen = prelen - 1
        valid_links_idx = torch.arange(prelen, dtype=torch.long, device=links.device).unsqueeze(1) + \
                    torch.arange(translen, dtype=torch.long, device=links.device).unsqueeze(0) + 1
        invalid_idx_mask = valid_links_idx >= prelen
        valid_links_idx.masked_fill_(invalid_idx_mask, prelen)
        res = torch.zeros(batch_size, prelen, prelen + 1, dtype=torch.float, device=links.device).fill_(float("-inf"))
        res.scatter_(2, valid_links_idx.unsqueeze(0).expand(batch_size, -1, -1), links)
        return res[:, :, :prelen]

    def initialize_output_tokens_with_length(self, src_tokens, length_tgt):
        r"""
        Function that initialize the decoder input with given length
        """
        max_length = length_tgt.max()
        idx_length = new_arange(src_tokens, max_length)

        initial_output_tokens = src_tokens.new_zeros(
            src_tokens.size(0), max_length
        ).fill_(self.pad) 
        r"""src_tokens.size(0) = batch size; initial_output_tokens.shape=[104,312]"""
        initial_output_tokens.masked_fill_(
            idx_length[None, :] < length_tgt[:, None], self.unk #TODO 
        ) # self.unk = 3 unknown token
        r"""
        idx_length[None, :].shape = [1, 312]
        length_tgt[:, None].shape = [104, 1]
        """
        initial_output_tokens[:, 0] = self.bos # self.bos=0
        initial_output_tokens.scatter_(1, length_tgt[:, None] - 1, self.eos) # self.eos=1
        r"""
        initial tokens = [[<bos>, <pad>, <pad>, ..., <pad>, <eos>], ...]; shape=[batch, max_length_target(lambda*max_input_length)]
        """
        return initial_output_tokens 

    def initialize_output_tokens_upsample_by_tokens(self, src_tokens):
        """
        src_tokens.shape = [batch, length]
        """
        length_tgt = torch.sum(src_tokens.ne(self.pad), -1) # count tokens that are not equal to pad
        length_tgt = (length_tgt * self.gen_decoder_input_args.upsample_scale).long().clamp_(min=2, max=1000) # upsample scale is 8 (i.e., lambda=8); custom max=1024 (matching positional embeddings)
        return self.initialize_output_tokens_with_length(src_tokens, length_tgt)
    
    def initialize_output_tokens_by_copy(self, src_tokens):
        #TODO come up with a way to give T5 decoder sensible inputs
        batch_size, padded_length = src_tokens.shape
        length_tgt = torch.sum(src_tokens.ne(self.pad), -1) # count tokens that are not equal to pad
        decoder_input_ids = torch.zeros((batch_size, min(512, torch.max(length_tgt)*5)))
        length_tgt = torch.sum(src_tokens.ne(self.pad), -1)
        pass



    
    def glat_step(self, node_word_logits, links, tgt_tokens, decoder_input_ids, enc_hidden_states, enc_attention_mask):
        batch_size, prelen, _ = links.shape
        tarlen = tgt_tokens.shape[1]
        nonpad_positions = ~tgt_tokens.eq(self.pad)
        target_length = (nonpad_positions).sum(1)
        output_length = decoder_input_ids.ne(self.pad).sum(1)

        pred_tokens = node_word_logits.argmax(-1)

        node_word_log_prob_on_tgt_tokens = torch_dag_logsoftmax_gather_inplace(node_word_logits, tgt_tokens.unsqueeze(1).expand(-1, prelen, -1))
        node_word_log_prob_on_tgt_tokens = node_word_log_prob_on_tgt_tokens.transpose(1, 2)

        if links.shape[-2] != links.shape[-1]:
            raise RuntimeError("links must have equal dimension in the -1 and -2 dimensions")
            # links = self.restore_valid_links(links)
        path = torch_dag_best_alignment(
            node_word_log_prob_on_tgt_tokens,
            links,
            output_length,
            target_length,
        )

        predict_align_mask = path >= 0
        matchmask = torch.zeros(batch_size, tarlen + 1, prelen, device=node_word_log_prob_on_tgt_tokens.device, dtype=torch.bool).scatter_(1, path.unsqueeze(1) + 1, 1)[:, 1:]
        oracle = tgt_tokens.gather(-1, path.clip(min=0)) # bsz * prelen
        same_num = ((pred_tokens == oracle) & predict_align_mask).sum(1)

        prob = torch.randn(oracle.shape, device=tgt_tokens.device, dtype=torch.float)
        prob.masked_fill_(~predict_align_mask, -100)
        glance_nums = ((target_length - same_num) * self.glat_params.get('context_p', 0.5) + 0.5).to(torch.long)
        #prob_thresh = prob.topk(glance_nums.max().clip(min=1))[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
        prob_thresh = prob.sort(descending=True)[0].gather(-1, (glance_nums - 1).clip(min=0).unsqueeze(-1)).squeeze(-1)
        prob_thresh.masked_fill_(glance_nums == 0, 100)
        keep_prob = (prob >= prob_thresh.unsqueeze(-1)).to(prob.dtype)

        keep_word_mask = (torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < keep_prob).bool()

        glat_decoder_input_ids = decoder_input_ids.masked_fill(keep_word_mask, 0) + oracle.masked_fill(~keep_word_mask, 0)

        glat_node_word_logits, glat_links \
            = self.nat_decoder.extract_word_logits_and_links(
                glat_decoder_input_ids, 
                enc_hidden_states,
                enc_attention_mask,
            )
        
        glat_info = {
            'glat_acc': (same_num.sum() / target_length.sum()).detach(),
            'glat_context_p': self.glat_params['context_p'],
            'glat_keep': keep_prob.mean().detach(),
            'matchmask': matchmask,
            'keep_word_mask': keep_word_mask,
            'glat_decoder_input_ids': glat_decoder_input_ids,
        }

        
        return glat_node_word_logits, glat_links, glat_decoder_input_ids, glat_info

    def _compute_dag_loss(
        self,
        outputs, # node_word_logits
        output_masks,
        targets,
        target_masks,
        links,
        name='dag',
        factor=1.0,
        label_smoothing=0.0,
        match_mask=None,
        keep_word_mask=None,
    ):
        batch_size = outputs.shape[0]
        prelen = outputs.shape[1]
        tarlen = targets.shape[1]

        output_length = output_masks.sum(dim=-1)
        target_length = target_masks.sum(dim=-1)

        # if self.cfg.torch_dag_logsoftmax_gather:
        match_all = torch_dag_logsoftmax_gather_inplace(outputs, targets.unsqueeze(1).expand(-1, prelen, -1))
        match_all = match_all.transpose(1, 2)

        if match_mask is not None and not self.cfg.no_force_emit:
            glat_prev_mask = keep_word_mask.unsqueeze(1)
            match_all = match_all.masked_fill(glat_prev_mask, 0) + match_all.masked_fill(~match_mask, float("-inf")).masked_fill(~glat_prev_mask, 0).detach()
        nvalidtokens = output_masks.sum()

        # if self.cfg.torch_dag_loss:
        if links.shape[-1] != links.shape[-2]:
            links = self.restore_valid_links(links)
        loss_result = torch_dag_loss(match_all, links, output_length, target_length)

        invalid_masks = loss_result.isinf().logical_or(loss_result.isnan())
        loss_result.masked_fill_(invalid_masks, 0)
        invalid_nsentences = invalid_masks.sum().detach()

        loss = -(loss_result / target_length).mean()
        nll_loss = loss.detach()
        nsentences, ntokens = targets.shape[0], targets.ne(self.pad).sum()

        loss_nofactor = loss
        loss = loss * factor

        return {"loss_name": name, "loss": loss, "nll_loss": nll_loss,
                "loss_factor": factor, "ntokens": ntokens, "nvalidtokens": nvalidtokens, "nsentences": nsentences,
                "loss_nofactor": loss_nofactor, "invalid_nsentences": invalid_nsentences}

    def _compute_dag_loss_with_dp(
        self,
        node_token_emit_logits,
        labels,
        links,
        name='dag_loss',
        factor=1.0,
    ):
        device = node_token_emit_logits.device
        batch_size, L, _ = node_token_emit_logits.shape # L stands for graph size, i.e., number of nodes in the graph
        tgt_lengths = (~labels.eq(self.pad)).sum(1)
        M = torch.max(tgt_lengths) # maximum target length (assume max_length padding in labels)
        f = torch.zeros((batch_size, M, L), device=device)
        f[f==0] = float("-inf") # log-likelihood (llk)
        b_idx = torch.arange(batch_size)
        f[b_idx, 0, 0] = node_token_emit_logits[b_idx, 0,labels[b_idx, 0]]
        for i in range(1, M):
            for u in range(i, L):
                trans_log_prob = logsumexp(f[b_idx,i-1,:u]+links[b_idx,:u, u], dim=1)
                emit_log_prob = node_token_emit_logits[b_idx,u,labels[b_idx,i]].squeeze(-1)
                f[b_idx, i, u] = emit_log_prob + trans_log_prob
        dag_loss = -torch.mean(f[b_idx, tgt_lengths-1, L-1]/tgt_lengths) # batch_loss, negative log-likelihood
        return {
            "name": name,
            "loss": dag_loss,
            "factor": factor,
            "loss_withfactor": dag_loss*factor,
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        
        return:
        - token output distribution of each node (node_word_logits)
        - DAG transition probability (links)
        - loss

        N.B.
        The model is NOT responsible for decoding (i.e., generating the final output tokens)
        """
        # Encode if needed (training, first prediction pass)
        r"""
        STEP 1. encode the source sentence
        """
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.base_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            
        enc_hidden_states, encoder_out_mask = encoder_outputs[0], attention_mask

        r"""
        STEP 2. obtain decoder input tokens
        """
        decoder_input_ids = self.initialize_output_tokens_upsample_by_tokens(input_ids)

        # Decode
        r"""
        STEP 3. Generate the DAG graph and output token probability of each node (node_word_logits)
        STEP 4. Compute transition probabilities (links)
        """
        node_word_logits, links = self.nat_decoder.extract_word_logits_and_links(
            decoder_input_ids, 
            encoder_hidden_states=enc_hidden_states,
            encoder_attention_mask=encoder_out_mask,
        )

        r"""
        STEP 5. (training) Apply glancing training
        """
        glat_info = None
        if labels is not None and self.use_glat:
            node_word_logits, links, glat_decoder_input_ids, glat_info = self.glat_step(node_word_logits, links, labels, decoder_input_ids, enc_hidden_states, encoder_out_mask)

        r"""
        STEP 6. (training) Compute Loss
        """
        loss_info = None
        if labels is not None:
            loss_info = self._compute_dag_loss(
            node_word_logits, 
            decoder_input_ids.ne(self.pad),
            labels,
            labels.ne(self.pad),
            links,
            name='dag-loss',
        )

        output = EasyDict({
            'node_word_logits': node_word_logits,
            'links': links,
            'decoder_input_ids': decoder_input_ids,
        })
        if loss_info is not None:
            output.update(loss_info)
        if glat_info is not None:
            output.update(glat_info)
        return output
    