from transformers.configuration_utils import PretrainedConfig
from transformers import T5ForConditionalGeneration
from transformers.models.t5.configuration_t5 import T5Config

from typing import Optional, Tuple, Union




class DirectedAcyclicTransformerConfig(PretrainedConfig):
    def __init__(
        self,
        base_model_class='T5',
        base_model_version='t5-small',
        link_features="feature:position",
        gen_decoder_input_args={'type':'upsample', 'upsample_scale':8},
        use_glat=True,
        glat_params={'context_p': 0.5},
        reinit_lm_head=True,
        d_model=512,
        vocab_size=30000,
        use_pretrained_base=False,
        use_pretrained_encoder=True,
        **kwargs
    ):
        self.base_model_class = base_model_class
        self.base_model_version = base_model_version
        self.link_features = link_features
        self.gen_decoder_input_args = gen_decoder_input_args
        self.use_glat = use_glat
        self.glat_params = glat_params
        self.reinit_lm_head = reinit_lm_head
        self.use_pretrained_base = use_pretrained_base
        self.use_pretrained_encoder = use_pretrained_encoder

        self.base_config = T5Config.from_pretrained(self.base_model_version)

        super().__init__(
            pad_token_id=self.base_config.pad_token_id,
            eos_token_id=self.base_config.eos_token_id,
            is_encoder_decoder=self.base_config.is_encoder_decoder,
            **kwargs
        )
    


