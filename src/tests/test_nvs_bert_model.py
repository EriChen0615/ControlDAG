import sys
sys.path.append('src/')
sys.path.append('.')
from models.modeling_nvs_bert import NVSBert
from runway_for_ml.utils.util import get_tokenizer
import torch

if __name__ == '__main__':
    model = NVSBert(
        bert_model_version='bert-base-uncased',
        # lambda_thresh=0.9,
        lambda_p=1000,
        top_k=500,
    )

    tokenizer_config = {
        'version_name': 'bert-base-uncased',
        'class_name': 'BertTokenizerFast',
        'tokenize_kwargs': {
            'padding': 'max_length',
            'truncation': True
        },
    }
    tokenizer = get_tokenizer(tokenizer_config)
    tokenized_input = tokenizer(
       ["Hi! I am Eric."]
    )
    input_ids, attention_mask = torch.tensor(tokenized_input['input_ids'], dtype=torch.long), torch.tensor(tokenized_input['attention_mask'], dtype=torch.long)
    
    
    selected_vocab_idx = model(input_ids=input_ids, attention_mask=attention_mask)['selected_vocab_idx']
    selected_vocab = tokenizer.batch_decode(selected_vocab_idx)
    print(selected_vocab)
    pass
    