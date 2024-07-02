import pytest
# from transformers import T5Tokenizer
import sys
import json
sys.path.append('src/')
from utilities.vocab_trie import VTrie

def test_add_word():
    t = VTrie()
    t.add_word(['he'])
    t.add_word(['he', 'llo'])
    assert 'he' in t.root.children
    assert 'llo' in t.root.children['he'].children

def test_check_advance():
    t = VTrie()
    t.add_word(['he'])
    t.add_word(['he', 'llo'])
    t.add_word(['he', 'lp'])
    assert t.check_advance('he')
    assert not t.check_advance('hi')

def test_advance():
    t = VTrie()
    t.add_word(['he'])
    t.add_word(['he', 'llo'])
    t.advance('he')
    assert len(t.active_path) == 2
    assert 'llo' in t.active_path[-1].children
    assert t.active_path[-1].token == 'he'

def test_pop():
    t = VTrie()
    t.add_word(['he'])
    t.add_word(['hello'])
    t.advance('he')
    t.pop()
    assert len(t.active_path) == 1
    assert t.active_path[-1].token == ""

def test_is_word():
    t = VTrie()
    t.add_word(['he'])
    t.add_word(['he', 'llo', 'ween'])
    t.advance('he')
    assert t.is_word()
    t.advance('llo')
    assert not t.is_word()

def test_init():
    all_vocab_in_tokens = [['▁hello'], ['▁hello', 'we', 'en'], ['▁beautiful'], ['▁beauty']]
    t = VTrie(all_vocab=all_vocab_in_tokens)
    t.advance('▁hello')
    assert t.check_advance('we')

def test_check_sentence():
    all_vocab_in_tokens = [['▁hello'], ['▁hello', 'we', 'en'], ['▁Her'], ['▁phone'], ['▁number'], ['▁is'], ['▁07', '52', '-9', '93', '-22', '88'], ['▁Sure']] 

    tokenized_sentence1 = ['▁Sure', '.', '▁Her', '▁phone', '▁number', '▁is', '▁07', '52', '-9', '93', '-22', '88', '.', '▁hello', '.']
    tokenized_sentence2 = ['▁Sure', '▁of', '▁course', '.',  '▁Her', '▁phone', '▁number', '▁is', '▁07', '52', '-9', '93', '-22', '88', '.']
    tokenized_sentence3 = ['▁hello', 'we', '.', '▁Her', '▁phone', '▁number', '▁is', '▁07', '52', '-9', '93', '-22', '88', '.']
    tokenized_sentence4 = ['▁hello', 'we', 'en', '.', '▁Her', '▁phone', '▁number', '▁is', '▁07', '52', '-9', '93', '-22', '88', '.']

    t = VTrie(all_vocab=all_vocab_in_tokens)
    res1 = t.check_sentence(tokenized_sentence1)
    # t.reset()
    res2 = t.check_sentence(tokenized_sentence2)
    t.reset()
    res3 = t.check_sentence(tokenized_sentence3)
    t.reset()
    res4 = t.check_sentence(tokenized_sentence4)
    t.reset()


    assert res1 == True
    assert res2 == False
    assert res3 == False
    assert res4 == True

VOCAB_FILE = '/home/jc2124/rds/hpc-work/DAG-NLG-runway/tmp/v3_vtrie_allowed_vocab_tokens.json'
def test_cases():
    all_vocab = None
    with open(VOCAB_FILE, 'r') as f:
        all_vocab = json.load(f)
    vtrie = VTrie(all_vocab=all_vocab)

    tokenized_sentences = [
        ['▁How', '▁about', '▁Who', '▁You', '▁Are', '▁Are', '▁Je', 's', 'sie', '▁J', '▁You', '▁by', '▁Ma', 'mm', 'a', '▁Know', 's', '▁Best', '▁from', '▁the', '▁album', '▁Mas'],
        ['▁Con', 'firm', 'ing', '▁', 'a', '▁standard', '▁car', '▁from', '▁The', 's', '▁Union', '▁Station', '▁on', '▁March', '▁9', 'th', '▁at', '▁11', '▁am', '▁to', '▁March', '▁14', 'th', '.'],
        ['▁Please', '▁confirm', ':', '▁Booking', '▁', 'a', '▁3', ':', '45', '▁pm', '▁at', '▁Cook', 'ing', '▁alarm', 'ing', '▁at', '▁3', ':', '45', '▁pm', '.'],
    ]

    gt_results = [
        False,
        False,
        False,
    ]

    ans = []


    for tok_sen, gt in zip(tokenized_sentences, gt_results):
        res = vtrie.check_sentence(tok_sen)
        ans.append(res)

    assert ans == gt_results

    

    

# def test_from_vocab_list():
#     tokenizer = T5Tokenizer.from_pretrained('t5-small')
#     t = VTrie.from_vocab_list(['he', 'hello'], tokenizer)
#     assert '▁he' in t.root.children
    # assert 'hello' in t.root.children['he'].children
