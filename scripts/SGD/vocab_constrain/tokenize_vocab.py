from transformers import T5TokenizerFast
import json
import pynini
import os

def _build_symbol_table(tokenizer):
    symbol_table = pynini.SymbolTable()
    symbol_table.add_symbol('<epsilon>')

    vocab_list = list(sorted(tokenizer.vocab.items(), key=lambda x: x[1]))
    for token_token_id in vocab_list:
        (token, token_id) = token_token_id
        symbol_id = symbol_table.add_symbol(token)
        assert symbol_id == token_id + 1 # off-by-one due to <epsilon>=0
    return symbol_table

def _build_vocab_fsa(all_allowed_vocab_tokens, symbol_table):
    all_allowed_vocab_tokens = [" ".join(vocab_tokens) for vocab_tokens in all_allowed_vocab_tokens]
    all_vocab_fsa = pynini.union(*all_allowed_vocab_tokens, token_type=symbol_table).closure()
    return all_vocab_fsa

VOCAB_FILE_PATH = 'data/dstc8-schema-guided-dialogue/train_vocab.json'
if __name__ == '__main__':
    all_vocab = None
    with open(VOCAB_FILE_PATH, 'r') as f:
        all_vocab = json.load(f)
    
    t5_tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    encoded_vocab_ids = t5_tokenizer.batch_encode_plus(all_vocab, add_special_tokens=False)['input_ids']
    print(encoded_vocab_ids)
    t5_vocab_tokens = []
    for input_ids in encoded_vocab_ids:
        t5_vocab_tokens.append(t5_tokenizer.convert_ids_to_tokens(input_ids))
    print(t5_vocab_tokens)

    symbol_table = _build_symbol_table(t5_tokenizer)
    vocab_fsa = _build_vocab_fsa(t5_vocab_tokens)
    pass

 
    
