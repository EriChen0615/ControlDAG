from transformers import T5Tokenizer, T5TokenizerFast
from tokenizers.processors import TemplateProcessing

if __name__ == '__main__':
    sentence = "Hi! I am Eric!"
    t5_tokenizer_slow = T5Tokenizer.from_pretrained(
        't5-small',
    )
    t5_tokenizer_slow.add_tokens(['<s>'])
    template = TemplateProcessing(
        single="<s> $0 </s>",
        special_tokens=[('<s>', t5_tokenizer_slow._convert_token_to_id("<s>")),("</s>", 1)]
    )
    t5_tokenizer_slow.post_processor = template
    output = t5_tokenizer_slow(sentence)
    print(output)
    print(t5_tokenizer_slow.decode(output['input_ids']))
    pass