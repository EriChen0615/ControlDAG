import transformers
import torch
import datasets

def get_tokenizer(tokenizer_config):
    tokenizer_dict = tokenizer_config
    tokenizer_name = tokenizer_dict['version_name']
    tokenizer_class = tokenizer_dict['class_name']
    tokenizer_class_obj = getattr(transformers, tokenizer_class)
    from_pretrained_kwargs = tokenizer_config.get('from_pretrained_kwargs', {})
    tokenizer = tokenizer_class_obj.from_pretrained(tokenizer_name, **from_pretrained_kwargs)
    if tokenizer_class[:4] == 'GPT2':
        # special_tokens.update({'pad_token': '[PAD]'})
        tokenizer.pad_token = '[PAD]'
        # tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_tokens(tokenizer_dict.get('additional_tokens', []))
    return tokenizer

def batch_depad(x, attension_mask=None, y=None, pad_len=0):
    max_in_length = torch.max((~(x==0)).sum(dim=-1))+pad_len
    attension_mask = attension_mask[:, :max_in_length]
    max_out_length = torch.max((~(y==0)).sum(dim=-1))+pad_len
    return x[:, :max_in_length], attension_mask, y[:, :max_out_length]

def cast_ds_column_to_image(ds, columns):
    for col in columns:
        if col in ds.info.features:
            ds = ds.cast_column(col, datasets.features.image.Image())
        else:
            print(f"Feature {col} not in {ds}")
    return ds

