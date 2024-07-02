import string
import json
import os
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

def _get_all_vocab(dataset_split, strip_punct=True, no_numeric=True):
    all_vocab = defaultdict(int)
    sv_vocab = defaultdict(int)
    for instance in tqdm(dataset_split, desc='iterating split...'):
        # gt_text = instance['target'].translate(str.maketrans('', '', string.punctuation))
        gt_text = instance['target']

        slot_value_text = []
        for action in instance['dialog_acts']:
            slot_value_text.extend(action['values'])
        slot_value_text = " ".join(slot_value_text)
        # slot_value_text.translate(str.maketrans('', '', string.punctuation)) // uncomment to get v2. Note: this is no-op because it's NOT in-place

        words = gt_text.split(' ')
        for word in words:
            if strip_punct: # v4
                word = word.strip(string.punctuation) # strip punctuations
            if no_numeric and len(word) and word[0].isnumeric(): # v4
                continue
            elif len(word) > 0:
                all_vocab[word] += 1
        
        sv_words = slot_value_text.split(' ')
        for sv_word in sv_words:
            # if not sv_word.isnumeric():
            if strip_punct: # v4
                sv_word = sv_word.strip(string.punctuation) # v3 
            if no_numeric and len(word) and word[0].isnumeric(): # v4
                continue
            if len(sv_word) > 0: # v3
                sv_vocab[sv_word] += 1
        
    return all_vocab, sv_vocab

def get_schema_vocab(schema_file):
    vocab = set()
    with open(schema_file, 'r') as f:
        schema_list = json.load(f)
        for service in schema_list:
            service_name = service['service_name']
            # if service_name == 'Music_3':
            #     breakpoint()
            for slot in service['slots']:
                words_in_name = slot['name'].split('_')
                for word in words_in_name:
                    vocab.add(word)
                    vocab.add(word.lower())
                desc = slot['description']
                words_in_desc = desc.split(' ')
                for word in words_in_desc:
                    vocab.add(word)
                    vocab.add(word.lower())
    return vocab

VOCAB_FILE_PATH = 'data/dstc8-schema-guided-dialogue'
SCHEMA_FILES = [
    "data/schemas/train/schema.json",
    "data/schemas/test/schema.json",
    "data/schemas/dev/schema.json",
]
CUTOFF_PERCENTAGE = 100 

if __name__ == '__main__':
    
    ds = load_dataset('GEM/schema_guided_dialog')
    train_ds = ds['train']
    test_ds = ds['test']
    valid_ds = ds['validation']

    schema_vocab = set()
    for schema_file in SCHEMA_FILES:
        print(schema_file)
        schema_vocab.update(get_schema_vocab(schema_file))
    schema_vocab_list = list(schema_vocab)
    # breakpoint()

    train_vocab, train_sv_vocab = _get_all_vocab(train_ds)
    train_sv_vocab_list = [w for w, wc in train_sv_vocab.items()]
    sorted_words_and_counts = sorted([(w, wc) for w, wc in train_vocab.items()], key=lambda x: x[1], reverse=True)
    wc_df = pd.DataFrame({
        'word': [item[0] for item in sorted_words_and_counts],
        'count': [item[1] for item in sorted_words_and_counts]
    })
    wc_df['cumulative_percentage'] = 100.0 * wc_df['count'].cumsum()/wc_df['count'].sum()
    train_vocab_list = wc_df[wc_df['cumulative_percentage'] < CUTOFF_PERCENTAGE]['word'].tolist()
    train_vocab_list.extend(schema_vocab_list)
    print("Total number of vocabulary in train split:", len(train_vocab_list))
    # with open(os.path.join(VOCAB_FILE_PATH, f'train_vocab-{CUTOFF_PERCENTAGE}%-v6.json'), 'w') as f:
    #     json.dump(train_vocab_list, f)
    print(f"Vocabulary saved to {VOCAB_FILE_PATH}")

    
    # test_vocab, test_sv_vocab = _get_all_vocab(test_ds, strip_punct=False, no_numeric=False) # v6
    test_vocab, test_sv_vocab = _get_all_vocab(test_ds) # v7
    # test_vocab_stripped, test_sv_vocab_stripped = _get_all_vocab(test_ds)
    test_sv_vocab_list = [w for w, wc in test_sv_vocab.items()]
    test_vocab_list = [w for w, wc in test_vocab.items()]
    print("Total number of vocabulary in test slot values:", len(test_sv_vocab_list))

    all_vocab_list = list(set(train_vocab_list + test_sv_vocab_list + schema_vocab_list))
    print("Total size of vocabulary", len(all_vocab_list))
    
    # with open(os.path.join(VOCAB_FILE_PATH, f'all_vocab-{CUTOFF_PERCENTAGE}%+test_sv-v7.json'), 'w') as f:
    #     json.dump(all_vocab_list, f)
    print(f"Vocabulary saved to {VOCAB_FILE_PATH}")

    valid_vocab, valid_sv_vocab = _get_all_vocab(valid_ds)
    valid_vocab_list = [w for w, wc in valid_vocab.items()]
    valid_sv_vocab_list = [w for w, wc in valid_sv_vocab.items()]

    total_vocab_list = list(set(train_vocab_list + train_sv_vocab_list + test_vocab_list + test_sv_vocab_list + valid_vocab_list + valid_sv_vocab_list + schema_vocab_list))
    with open(os.path.join(VOCAB_FILE_PATH, f'total_vocab-train+test+valid-v7.json'), 'w') as f:
        json.dump(total_vocab_list, f)
    print(f"Vocabulary saved to {VOCAB_FILE_PATH}")

    # print("Total number of vocabulary in test split:", len(test_vocab))
    
    # diff_vocab = test_vocab - test_sv_vocab - train_vocab
    # print("Number of vocabulary not in slot values vocab and train vocab", len(diff_vocab))
    # with open(os.path.join(VOCAB_FILE_PATH, 'train_test_diff_vocab.json'), 'w') as f:
    #     json.dump([vocab for vocab in diff_vocab], f)
    # print(f"Diff Vocabulary saved to {VOCAB_FILE_PATH}")