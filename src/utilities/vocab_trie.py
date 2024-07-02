import string

class VTrieNode:
    def __init__(self, token, eow=False):
        self.token = token
        self.eow = eow # is end of word

        self.children = dict()
    
    def __str__(self) -> str:
        return f"(token={self.token}, eow={self.eow})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
class VTrie:
    def __init__(self, all_vocab=None, sep_char='▁'):
        # self.root = VTrieNode("", False)
        self.root = VTrieNode("", True)
        self.active_path = [self.root]
        self.sep_char = sep_char

        if all_vocab is not None:
            for vocab in all_vocab:
                self.add_word(vocab)
    
    def __repr__(self) -> str:
        return "|".join([node.__repr__() for node in self.active_path])
    
    def add_word(self, subword_seq):
        cur_node = self.root
        word_len = len(subword_seq)
        word_added = False
        for i, subword in enumerate(subword_seq):
            if subword not in cur_node.children:
                cur_node.children[subword] = VTrieNode(subword)

            is_eow = (i==word_len-1)
            if is_eow:
                word_added = not cur_node.children[subword].eow
                cur_node.children[subword].eow = True
            
            cur_node = cur_node.children[subword]
        return word_added
    
    def remove_word(self, subword_seq):
        cur_node = self.root
        word_len = len(subword_seq)
        word_removed = False
        for i, subword in enumerate(subword_seq):
            if subword not in cur_node.children:
                cur_node.children[subword] = VTrieNode(subword)

            is_eow = (i==word_len-1)
            if is_eow:
                word_removed = cur_node.children[subword].eow
                cur_node.children[subword].eow = False
            
            cur_node = cur_node.children[subword]
        return word_removed
    
    def check_advance(self, token):
        """Conditions for advance:
        1. `token` in cur_node.children 
        2. Encounter a punctuation, and cur_node is word
        3. A new word has started (marked by a leading '▁' character)

        :param token: _description_
        :return: _description_
        """
        cur_node = self.active_path[-1]
        advancable = False
        if token in cur_node.children:
            advancable = True
        elif token in string.punctuation and cur_node.eow:
            advancable = True
        # elif token[0] == self.sep_char and (cur_node.eow or cur_node.token in string.punctuation) and token in self.root.children:
        elif token[0] == self.sep_char and (cur_node.eow) and token in self.root.children:
            advancable = True
        return advancable
    
    def advance(self, token):
        cur_node = self.active_path[-1]
        if token in cur_node.children:
            self.active_path.append(cur_node.children[token])
        elif token in string.punctuation and cur_node.eow:
            self.active_path.append(self.root)
        elif token[0] == self.sep_char and (cur_node.eow) and token in self.root.children:
            self.active_path.append(self.root.children[token])
        else:
            raise RuntimeError(f"Cannot advance {token}. Make sure you called `check_advance` before `advance`")
    
    def reset(self):
        if len(self.active_path) == 1: # already reset. This happens when a punctuation is advanced
            return
        self.active_path = [self.root]
    
    def check_sentence(self, tokenized_sentence, verbose=False):
        for subword in tokenized_sentence:
            if not self.check_advance(subword):
                if verbose:
                    print(f"Failed at {subword}")
                self.reset()
                return False
            else:
                self.advance(subword)
        res = self.is_word()
        self.reset()
        return res

    def pop(self):
        self.active_path.pop()
    
    def is_word(self):
        cur_node = self.active_path[-1]
        return cur_node.eow
    
    @classmethod
    def from_vocab_list(cls, vocab_list, tokenizer=None):
        encoded_vocab_ids = tokenizer.batch_encode_plus(vocab_list, add_special_tokens=False)['input_ids']
        all_vocab_in_tokens = []
        for input_ids in encoded_vocab_ids:
            all_vocab_in_tokens.append(tokenizer.convert_ids_to_tokens(input_ids))
        
        vt = cls(all_vocab=all_vocab_in_tokens)
        return vt


            

