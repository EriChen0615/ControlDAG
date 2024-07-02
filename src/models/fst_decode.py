import torch
import pynini
from transformers import T5TokenizerFast
from runway_for_ml.utils.util import get_tokenizer
from runway_for_ml.utils.eval_recorder import EvalRecorder
from data_ops.eval_ops import ComputeNeologismRate
import time
import json
import pathlib
import os
import numpy as np
from collections import defaultdict
from utilities.vocab_trie import VTrie
from utilities.utils import torch_tensor_intersect, unique_rows_indices
import copy
import math

import logging
logger = logging.getLogger(__name__)

log_formatter = logging.Formatter(fmt='%(name)s :: [%(levelname)s] :: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(log_formatter)

logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)


class FSTDecoder:
    def __init__(
        self, 
        tokenizer_config, 
        top_k_transitions=2, 
        top_k_emissions=3, 
        use_constraints=True, 
        apply_vocab_constraint="fsa", # "fsa" or "trie" or None (False) 
        vocab_file=None, 
        use_cache_fst_file=True, 
        add_vocab_dynamically=True, 
        word_insertion_penalty=0.0, 
        add_wip_to_start_of_word=True,
        len_constraint_ranking='min_norm_dist_with_penalty',  # "exact_length", ...
        len_search_algo='none', # dfs_memo, bfs, top_search
        len_strictness_A=1.0,
        len_prune_cumprob=0.7,
        eval_neologism_on_the_fly=False,
        add_forced_tokens_to_dag=True,
        dag_vocab_constraint=None,
        num_enforced_first_sv_tokens=0.3,
    ):
        self._build_symbol_table(tokenizer_config) # build self.tokenizer and self.symbol_table
        self.match_all_wildcard = self._build_match_all_wildcard()
        self.top_k_transitions = top_k_transitions
        self.top_k_emissions = top_k_emissions
        self.use_constraints = use_constraints
        self.add_vocab_dynamically = add_vocab_dynamically
        self.word_insertion_penalty = word_insertion_penalty
        self.add_wip_to_start_of_word = add_wip_to_start_of_word

        self.apply_vocab_constraint = apply_vocab_constraint
        self.vocab_file = vocab_file
        self.use_cache_fst_file = use_cache_fst_file

        self.length_constraint_fail_cnt = 0
        self.len_constraint_ranking = len_constraint_ranking
        self.len_search_algo = len_search_algo
        self.len_strictness_A = len_strictness_A
        self.len_prune_cumprob = len_prune_cumprob

        self.eval_neologism_on_the_fly = eval_neologism_on_the_fly
        self.add_forced_tokens_to_dag = add_forced_tokens_to_dag
        self.dag_vocab_constraint = dag_vocab_constraint
        self.num_enforced_first_sv_tokens = num_enforced_first_sv_tokens

        logger.debug(f"Word insertion penalty = {word_insertion_penalty}")
        logger.debug(f"Use constaints: {self.use_constraints}")
        logger.debug(f"Apply vocabulary constraint: {self.apply_vocab_constraint}")
        logger.debug(f"Adding vocabulary dynamically: {self.add_vocab_dynamically}")
        logger.debug(f"Length Constraint ranking method: {self.len_constraint_ranking}")
        logger.debug(f"DAG Vocabulary Constarint: {self.dag_vocab_constraint}")
        logger.debug(f"Number of enforced first sv tokens: {self.num_enforced_first_sv_tokens}")
        print(f"Length Constraint ranking method: {self.len_constraint_ranking}")
        print(f"Length Constraint Search algorithm: {self.len_search_algo}")
        print(f"(If DFS memo) Length Constraint Parameters: A={self.len_strictness_A}, Prune CUMPROB={self.len_prune_cumprob}")

        # Get all allowed bi-grams
        if self.apply_vocab_constraint =='bigram':
            with open(self.vocab_file, 'r') as f:
                self.all_allowed_vocab = json.load(f)    
            encoded_vocab_ids = self.tokenizer.batch_encode_plus(self.all_allowed_vocab, add_special_tokens=False)['input_ids']

            self.all_allowed_bigrams = defaultdict(set)
            self.all_end_tokens = set()
            self.all_start_tokens = set()
            for input_ids in encoded_vocab_ids:
                if len(input_ids) == 0: continue
                self.all_start_tokens.add(input_ids[0])
                self.all_end_tokens.add(input_ids[-1])
                for idx in range(len(input_ids)-1):
                    self.all_allowed_bigrams[input_ids[idx]].add(input_ids[idx+1])
            for k, v in self.all_allowed_bigrams.items():
                self.all_allowed_bigrams[k] = list(v)
            self.all_start_tokens = list(self.all_start_tokens)
            self.all_allowed_bigrams[32100] = [i for i in range(32128)] # IMPORTANT: all tokens are valid after <s>
            self.all_punct_token_ids = set(self.tokenizer.convert_tokens_to_ids([p for p in '!$&\'()*+,-./:;=>?@[]_']))
            logger.debug(f"Total number of allowed bigrams: {sum([len(v) for v in self.all_allowed_bigrams.values()])}")

            if logger.getEffectiveLevel() <= logging.DEBUG: 
                dict_to_show = {self.tokenizer.convert_ids_to_tokens([t1])[0]: self.tokenizer.convert_ids_to_tokens(ts) for i, (t1, ts) in enumerate(self.all_allowed_bigrams.items()) if i < 10}
                logger.debug(f"All allowed bigrams: {dict_to_show}")
                breakpoint()

        if self.apply_vocab_constraint == 'fsa': # enforce all vocabulary (i.e., 1-gram) are contained in the vocabulary file specified.
            self.fst_file_path = ''.join(self.vocab_file.split('.')[:-1]+['.fst'])
            # self.fst_file_path = str(self.fst_file_path.rename(self.fst_file_path.with_suffix('.fst')))
            if self.use_cache_fst_file and os.path.exists(self.fst_file_path):
                self.unclosed_all_allowed_vocab_acceptor = pynini.Fst.read(self.fst_file_path)
                self.all_allowed_vocab_acceptor = self.unclosed_all_allowed_vocab_acceptor.closure()
                logger.debug(f"Using cached FST file at {self.fst_file_path}")
            else:
                self.all_allowed_vocab = None
                with open(self.vocab_file, 'r') as f:
                    self.all_allowed_vocab = json.load(f)
                logger.debug(f"Read vocabulary from file {self.vocab_file}. Total size = {len(self.all_allowed_vocab)}]")
                encoded_vocab_ids = self.tokenizer.batch_encode_plus(self.all_allowed_vocab, add_special_tokens=False)['input_ids']

                allowed_vocab_tokens = [] # e.g. [['▁Lewis'], ['▁W', 'edge', 'wood'], ['▁Now']]
                for input_ids in encoded_vocab_ids:
                    allowed_vocab_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids))

                # add the numeric vocabulary that was removed
                # numeric_vocab = [[vocab] for vocab in self.tokenizer.vocab if vocab.isnumeric() or vocab[1:].isnumeric()]  # add all numeric tokens in vocabulary
                numeric_vocab = [[vocab] for vocab in self.tokenizer.vocab if (len(vocab) and vocab[0] == '▁' and vocab[1:].isnumeric())] # only add standalone digits
                dollar_vocab = [[vocab] for vocab in self.tokenizer.vocab if vocab.startswith('▁$')]
                # print('numeric_vocab', numeric_vocab)
                # print('dollar_vocab', dollar_vocab)
                # input("(BREAKPOINT)")
                allowed_vocab_tokens.extend(numeric_vocab)
                allowed_vocab_tokens.extend(dollar_vocab)
                # add BOS/EOS special token, punctuations, and blank 
                allowed_vocab_tokens.extend([["<s>"], ["</s>"]])
                allowed_vocab_tokens.extend([[p] for p in '!$&\'()*+,-./:;=>?@[]_'])
                allowed_vocab_tokens.append(['▁'])
                logger.debug(f"Total number of allowed vocabulary: {len(allowed_vocab_tokens)}")

                self.all_allowed_vocab_acceptor, self.unclosed_all_allowed_vocab_acceptor = self._build_allowed_vocab_fsa(allowed_vocab_tokens)

                opt_start_time = time.time()
                self.all_allowed_vocab_acceptor = pynini.optimize(self.all_allowed_vocab_acceptor)
                self.unclosed_all_allowed_vocab_acceptor = pynini.optimize(self.unclosed_all_allowed_vocab_acceptor)
                opt_end_time = time.time()
                logger.debug(f"Optimizing the vocabulary acceptor took {opt_end_time-opt_start_time} seconds")
                self.unclosed_all_allowed_vocab_acceptor.write(self.fst_file_path)
                logger.debug(f"Caching optimized FST (unclosed) to {self.fst_file_path}")
                
                

        elif self.apply_vocab_constraint == 'trie':
            assert self.len_search_algo == 'dfs_memo', f'Trie-based vocabulary constraint only works with DFS_memo graph search algorithm, but len_search_algo={len_search_algo}'
            with open(self.vocab_file, 'r') as f:
                self.all_allowed_vocab = json.load(f)
            encoded_vocab_ids = self.tokenizer.batch_encode_plus(self.all_allowed_vocab, add_special_tokens=False)['input_ids']

            self.allowed_vocab_tokens = [] # e.g. [['▁Lewis'], ['▁W', 'edge', 'wood'], ['▁Now']]
            for input_ids in encoded_vocab_ids:
                self.allowed_vocab_tokens.append(self.tokenizer.convert_ids_to_tokens(input_ids))
            # token_vocab_json = 'tmp/v3_vtrie_allowed_vocab_tokens.json' 
            # with open(token_vocab_json, 'w') as f:
            #     json.dump(self.allowed_vocab_tokens, f, ensure_ascii=False)
            #     input(f"(BREAKPOINT) Dumped to {token_vocab_json}")
            self.vtrie = VTrie(all_vocab=self.allowed_vocab_tokens)
            for special_word in [["<s>"], ["</s>"]]:
                self.vtrie.add_word(special_word)
        
        elif self.apply_vocab_constraint is not None and self.apply_vocab_constraint is not False:
            raise NotImplementedError(f"vocabulary constraint method: {self.apply_vocab_constraint} is not implemented")
            
            # print([node for _, node in self.vtrie.root.children.items()])
            # input("(BREAKPOINT) inspect vtrie root node")
        
        if self.eval_neologism_on_the_fly:
            self.compute_neologism_op = ComputeNeologismRate()
            self.compute_neologism_op.setup(all_vocab_file=self.vocab_file, lower_case=False, no_numeric=True, strip_punct=True, save_to_file=False)
    
    def _prune_dag(self, node_word_logits, links, forced_token_ids=None, top_k=3, add_forced_tokens=True, top_k_links=3, dag_vocab_constraint=None):
        """Note: only works for batch_size=1 because the way forced_token_ids is set

        Args:
            node_word_logits (_type_): _description_
            links (_type_): _description_
            forced_token_ids (_type_): _description_
            top_k (int, optional): _description_. Defaults to 3.
        """
        # DEBUG
        logger.debug(f"In _prune_dag() function: top_k={top_k}, add_foced_tokens={add_forced_tokens}, dag_vocab_constraint={dag_vocab_constraint}")
        logger.debug(f"forced_token_ids = {forced_token_ids}")
        logger.debug(f"top_k = {top_k}")
        logger.debug(f"node_word_logits.shape = {node_word_logits.shape}")
        # if logger.getEffectiveLevel() <= logging.DEBUG:
        #     breakpoint()
        
        # batch_forced_token_ids = []
        all_forced_token_ids = []
        if add_forced_tokens and forced_token_ids is not None and len(forced_token_ids):
            for t_ids in forced_token_ids[0]:
                all_forced_token_ids.extend(t_ids)

        _, top_idx = torch.topk(node_word_logits, top_k, dim=-1)

        batch_size, graph_size, _ = node_word_logits.shape

        select_idx = torch.zeros((batch_size, graph_size, top_k+len(all_forced_token_ids)), dtype=torch.long).to(node_word_logits.device)

        select_idx[:,:,:top_k] = top_idx

        # Option 2. force continuation. Get top links
        _, next_node = torch.topk(links, top_k_links, dim=-1)
        next_node = next_node[0] # batch_size=1
        if dag_vocab_constraint == 'bigram':
            if logger.getEffectiveLevel() <= logging.DEBUG:
                prev_top_idx = copy.deepcopy(top_idx)
            prev_nodes = defaultdict(set) 
            for this_node in range(node_word_logits.shape[1]):
                for to_node in next_node[this_node]:
                    to_node = to_node.item()
                    if to_node > this_node:
                        prev_nodes[to_node].add(this_node)
                if len(prev_nodes[this_node]) == 0:
                    logger.debug(f"Node ID={this_node}. No prev nodes: skipping")
                    pass
                else:
                    all_tokens = []
                    for prev_node in prev_nodes[this_node]:
                        for emit_token_id in top_idx[0, prev_node, :].tolist():
                            all_tokens += list(self.all_allowed_bigrams[emit_token_id])
                            if emit_token_id in self.all_end_tokens or emit_token_id in self.all_punct_token_ids:
                                all_tokens += self.all_start_tokens
                                all_tokens += self.all_punct_token_ids
                    all_tokens = list(set(all_tokens))

                    vocab_mask = torch.zeros_like(node_word_logits[0, this_node, :], dtype=torch.bool).to(node_word_logits.device)
                    vocab_mask[all_tokens] = True
                    masked_node_word_logits = node_word_logits[0, this_node, :].masked_fill(~vocab_mask, -float('inf'))
                    top_idx[0, this_node, :min(len(all_tokens), top_k)] = torch.topk(masked_node_word_logits, min(len(all_tokens), top_k), dim=-1)[1] 

                    if logger.getEffectiveLevel() <= logging.DEBUG: 
                        logger.debug(f"Node ID={this_node}. all_allowed_tokens={self.tokenizer.convert_ids_to_tokens(all_tokens)}")
                        logger.debug(f"Node ID={this_node}. prev_top_tokens={self.tokenizer.convert_ids_to_tokens(prev_top_idx[0, this_node])}")
                        logger.debug(f"Node ID={this_node}. prev_top_idx={prev_top_idx[0, this_node]}")
                        logger.debug(f"Node ID={this_node}. after vocab constraint top_tokens={self.tokenizer.convert_ids_to_tokens(top_idx[0, this_node])}")
                        logger.debug(f"Node ID={this_node}. after vocab constraint top_idx={top_idx[0, this_node]}")
                        breakpoint()


            if logger.getEffectiveLevel() <= logging.DEBUG:
                logger.debug(f"prev_top_idx = {prev_top_idx[0, :10, :]}")
                logger.debug(f"after vocab constraint top_idx = {top_idx[0, :10, :]}")
                breakpoint()

        if len(all_forced_token_ids) and add_forced_tokens == "naive":
            # Option 1. brute-force add all tokens
            select_idx[:,:,top_k:] = torch.tensor(all_forced_token_ids, dtype=torch.long).to(select_idx.device)
        elif len(all_forced_token_ids) and add_forced_tokens:
            # #TODO parallelize the computation
            all_non_last_sv_token_ids = []
            next_sv_token_ids = []
            next_sv_token_select_pos_offset = 0
            next_sv_token_select_pos_list = []
            prev_sv_token_mask = torch.zeros_like(node_word_logits, dtype=torch.bool).to(node_word_logits.device)
            for t_ids in forced_token_ids[0]:
                num_enforced_first_hlc_tokens = self.num_enforced_first_sv_tokens if isinstance(self.num_enforced_first_sv_tokens, int) else int(node_word_logits.shape[1]*self.num_enforced_first_sv_tokens)
                _, first_token_node_idx = torch.topk(node_word_logits[0, :, t_ids[0]], min(num_enforced_first_hlc_tokens, node_word_logits.shape[1]), dim=-1)
                select_idx[0, first_token_node_idx, top_k+next_sv_token_select_pos_offset] = t_ids[0]
                prev_sv_token_mask[0, first_token_node_idx, t_ids[0]] = True # these sv_tokens are activated. 
                all_non_last_sv_token_ids.extend(t_ids[:-1])
                next_sv_token_ids.extend(t_ids[1:])
                next_sv_token_select_pos_list.extend([top_k+i+next_sv_token_select_pos_offset for i in range(1, len(t_ids))])
                next_sv_token_select_pos_offset += len(t_ids)
            
            next_sv_token_ids = torch.tensor(next_sv_token_ids, dtype=torch.long, device=select_idx.device)
            prev_sv_token_mask.scatter_(-1, top_idx, True)
            prev_sv_token_mask = prev_sv_token_mask[0, : ,all_non_last_sv_token_ids] 
            next_sv_token_indicator = torch.zeros_like(prev_sv_token_mask, dtype=torch.bool, device=select_idx.device)

            #TODO: parallelize the computation below
            for tidx in range(prev_sv_token_mask.shape[-1]):
                next_sv_token_indicator[next_node[prev_sv_token_mask[:, tidx]], tidx] = True

            select_idx[0, :, next_sv_token_select_pos_list] = torch.where(next_sv_token_indicator, next_sv_token_ids.repeat((graph_size, 1)), 0)

            sv_offset = 0
            for t_ids in forced_token_ids[0]:
                for ii in range(len(t_ids[:-1])):
                    start_nodes_to_cont = select_idx[0, :, top_k+sv_offset+ii].nonzero().squeeze(-1)
                    nodes_to_cont = next_node[start_nodes_to_cont].flatten()
                    select_idx[0, nodes_to_cont, top_k+sv_offset+ii+1] = t_ids[ii+1]
                sv_offset += len(t_ids)


            # num_ftokens = prev_sv_token_mask.shape[-1]
            # col_ind = torch.arange(num_ftokens)
            # next_sv_token_indicator_prime = torch.zeros_like(prev_sv_token_mask, dtype=torch.bool).to(select_idx.device)
            # next_sv_token_indicator_prime = next_sv_token_indicator_prime.unsqueeze(1).repeat((1, num_ftokens, 1)) 
            # next_node_exp = next_node.unsqueeze(1).repeat((1, num_ftokens, 1))
            # next_sv_token_indicator_prime[next_node_exp[prev_sv_token_mask[:, col_ind], :], col_ind] = True
            # next_sv_token_indicator_prime = next_sv_token_indicator_prime.any(dim=1)


            # if logger.getEffectiveLevel() <= logging.DEBUG:
            #     breakpoint()

            # # Option 2. force continuation (naive implementation)
            # idx_offset = top_k
            # for t_ids in forced_token_ids[0]:
            #     # add first token emission
            #     _, first_token_node_idx = torch.topk(node_word_logits[0, :, t_ids[0]], min(self.num_enforced_first_sv_tokens, node_word_logits.shape[1]), dim=-1)
            #     select_idx[0, first_token_node_idx, idx_offset] = t_ids[0]
            #     idx_offset += 1
            #     for j in range(len(t_ids)-1):
            #         for i in range(graph_size):
            #             if t_ids[j] in select_idx[0, i, :idx_offset+j]:
            #                 # if logger.getEffectiveLevel() <= logging.DEBUG: breakpoint()
            #                 select_idx[0, next_node[i], idx_offset+j] = t_ids[j+1] # force continuation
            #     idx_offset += len(t_ids)-1

            # if not torch.all(select_idx == select_idx_copy): #NOTE for debugging
            #     print("forced_token_ids:", forced_token_ids)
            #     breakpoint()   # breakpoint()
            

        pruned_node_word_logits = node_word_logits.gather(-1, select_idx)
        node_token_ids = select_idx + 1 # to make up for the off-by-one due to epsilon

        logger.debug(f"select_idx = {select_idx[0, :10, :]}")
        
        if logger.getEffectiveLevel() <= logging.DEBUG:
            breakpoint()

        return pruned_node_word_logits, node_token_ids, links
        
    def _build_symbol_table(self, tokenizer_config):
        self.tokenizer = get_tokenizer(tokenizer_config)
        self.symbol_table = pynini.SymbolTable()
        self.symbol_table.add_symbol('<epsilon>')
        
        vocab_list = list(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1]))
        for token_token_id in vocab_list:
            (token, token_id) = token_token_id
            symbol_id = self.symbol_table.add_symbol(token)
            assert symbol_id == token_id + 1 # off-by-one due to <epsilon>=0
        pass
        
    def _build_match_all_wildcard(self):
        A = pynini.accep("", token_type=self.symbol_table)
        for i in range(1, self.symbol_table.available_key()): # do not add epsilon
            edge = (i, i, 0.0, 0)
            new_arc = pynini.Arc(*edge)
            A.add_arc(0, new_arc)
        return A
    
    def _build_domain_slot_fsa(self, domain_name, slot_name, valid_slot_values=None):
        #NOTE: A draft. Check implementation
        prefix = f"{domain_name} {slot_name} "
        prefix_fsa = pynini.accep(prefix, token_type=self.symbol_table)
        if valid_slot_values is None:
            slot_value_fsa = self.match_all_wildcard
        else:
            _, slot_value_fsa = self._build_allowed_vocab_fsa(valid_slot_values) 
        constraint_fsa = self.match_all_wildcard + prefix_fsa + slot_value_fsa + self.match_all_wildcard #NOTE: check if you need the wildcard in the front.
        return constraint_fsa
        
    
    def _build_allowed_vocab_fsa(self, all_allowed_vocab_tokens):
        """Must be called after symbol table is available

        :param allowed_vocab_tokens: _description_
        """
        all_allowed_vocab_tokens = [" ".join(vocab_tokens) for vocab_tokens in all_allowed_vocab_tokens]
        vocab_fsa_list = [pynini.accep(vocab, token_type=self.symbol_table) for vocab in all_allowed_vocab_tokens]
        unclosed_allowed_vocab_fsa = pynini.union(*vocab_fsa_list)
        return unclosed_allowed_vocab_fsa.closure(), unclosed_allowed_vocab_fsa
    
    def make_edges_naive(self, node_token_logits, node_token_idx, links, graph_lengths):
        batch_edges = []
        weight_mat = links[:, :, :, None] + node_token_logits[:, :, None, :]
        for batch_idx, (this_node_token_idx, this_links) in enumerate(zip(node_token_idx, links)):
            num_nodes, per_node_emission_num = this_node_token_idx.shape
            _, all_next_states = torch.topk(this_links, per_node_emission_num, dim=-1)
            this_edges = [] # item: (this_state, label, weight, next_state)
            for this_state_idx in range(num_nodes-1):
                next_states = all_next_states[this_state_idx]
                for next_state_idx in next_states:
                    for k in range(per_node_emission_num):
                        label = self.symbol_table.find(this_node_token_idx[this_state_idx, k])
                        weight = weight_mat[batch_idx, this_state_idx, next_state_idx, k]
                        edge = (this_state_idx, label, -weight.item()+self.word_insertion_penalty, next_state_idx.item()) # NOTE: negate weight because WFST operation are designed to MINIMIZE
                        this_edges.append(edge)
            batch_edges.append(this_edges)
        return batch_edges # indexed by batch idx   
    
    def make_wfsa_from_naive_edges(self, edges):
        wfsa = pynini.accep("", token_type=self.symbol_table)
        end_state = edges[-1][0] + 1
        # while wfsa.add_state() != end_state: continue # looks stupid to me... surely there should be alternative?
        wfsa.add_states(end_state+1)
        for edge in edges:
            start_state, label, weight, dest_state = edge
            label_symbol_id = self.symbol_table.find(label)
            new_arc = pynini.Arc(label_symbol_id, label_symbol_id, weight, dest_state)
            wfsa.add_arc(start_state, new_arc)
        wfsa.set_start(0)
        wfsa.set_final(0, float('inf'))
        wfsa.set_final(end_state, 0.0)
        return wfsa
    
    def make_edges(self, node_token_logits, node_token_idx, links, graph_lengths):
        """NOTE: only works for batch size = 1

        Args:
            node_token_logits (_type_): _description_
            node_token_idx (_type_): _description_
            links (_type_): _description_
            graph_lengths (_type_): _description_

        Returns:
            _type_: _description_
        """
       
        logger.debug(f"make_edges(): node_token_idx.shape={node_token_idx.shape}")
        # if logger.getEffectiveLevel() <= logging.DEBUG:
        #     breakpoint()
        weight_mat = links[:, :, :, None] + node_token_logits[:, :, None,:] 
        batched_edges = []
        device = links.device
        for batch_idx, (this_node_token_idx, this_links) in enumerate(zip(node_token_idx, links)):
            _, all_next_states = torch.topk(this_links, self.top_k_transitions, dim=-1)
            num_nodes, per_node_emission_num = this_node_token_idx.shape
            out_degree = per_node_emission_num * self.top_k_transitions
            num_edges = out_degree * (num_nodes - 1) # -1 due to no transition from final node
            
            start_node_col = torch.arange(num_nodes-1, dtype=torch.long).repeat_interleave(out_degree).to(device) # column for start token
            dest_node_col = all_next_states[:-1].repeat_interleave(out_degree // self.top_k_transitions, dim=-1) # -1: destination except for the last state
            dest_node_col = dest_node_col.flatten()
            weight_token_idx = torch.arange(per_node_emission_num).repeat(num_edges // per_node_emission_num).to(device) # used to index weights
            weights_col = weight_mat[batch_idx, start_node_col, dest_node_col, weight_token_idx] #NOTE: only works for batch_size = 1
            label_col = this_node_token_idx[:-1].repeat(1, out_degree // per_node_emission_num).flatten()
            
            # edges = torch.vstack([start_node_col[None:], label_col[None:], -weights_col[None:]+self.word_insertion_penalty, dest_node_col[None:]]).T
            edges = torch.vstack([start_node_col[None:], label_col[None:], -weights_col[None:], dest_node_col[None:]]).T
            logger.debug(f"Before dropping: edges.shape={edges.shape}")
            rows_to_drop = torch.logical_or(torch.logical_or(edges[:,0] >= edges[:,3], edges[:,2] > 1e5), edges[:, 1]==1) # drop cyclic transitions and weights to large edges AND those with [pad] (T5 token 0, vocab table 1) labels
            edges = edges[~rows_to_drop]
            logger.debug(f"After dropping: edges.shape={edges.shape}")

            # logger.debug(f"Before unique edges.shape={edges.shape}")
            # unique_indices = unique_rows_indices(edges[:, [0, 1, 3]])
            # edges = edges[unique_indices]
            # logger.debug(f"After unique edges.shape={edges.shape}")


            if logger.getEffectiveLevel() <= logging.DEBUG:
                breakpoint()

            batched_edges.append(edges)
        

        return batched_edges #NOTE: only works for batch_size = 1
    
    def make_wfsa_from_edges(self, edges, final_state=None):
        wfsa = pynini.accep("", token_type=self.symbol_table)
        end_state = final_state or edges[-1][0] + 1
        # while wfsa.add_state() != end_state: continue # looks stupid to me... surely there should be alternative?
        wfsa.add_states(end_state)
        reachable_states = set({0})
        for edge in edges.tolist():
            start_state, label_symbol_id, weight, dest_state = edge
            start_state, label_symbol_id, dest_state = int(start_state), int(label_symbol_id), int(dest_state)
            reachable_states.add(dest_state)
            if start_state in reachable_states: # because the edges are in topological order, any state not in the state are not reachable, and hence can be reduced.
                if (self.word_insertion_penalty < -0.0001 or self.word_insertion_penalty > 0.0001) and self.add_wip_to_start_of_word:
                    symb = self.symbol_table.find(label_symbol_id)
                    weight = weight + self.word_insertion_penalty if symb.startswith('▁') else weight
                new_arc = pynini.Arc(label_symbol_id, label_symbol_id, weight+self.word_insertion_penalty, dest_state)
                wfsa.add_arc(start_state, new_arc)
        wfsa.set_start(0)
        wfsa.set_final(0, float('inf'))
        wfsa.set_final(end_state, 0.0)
        logger.debug(f"DAG-to-WFSA WFSA size: {wfsa.num_states()}")
        if logger.getEffectiveLevel() <= logging.DEBUG: #TODO BUG TO BE FIXED! end state
            logger.debug(f"end_state={end_state}, final_state={final_state}, edges[-1][0]+1={edges[-1][0] + 1}")
            breakpoint()
        return wfsa

    def make_constrained_FSAs(self, forced_token_ids):
        A_cons = []
        for phrase_token_ids in forced_token_ids:
            constarined_phrase = " ".join(self.tokenizer.convert_ids_to_tokens(phrase_token_ids))
            # print(constarined_phrase)
            # input("(BREAKPOINT)")
            A = pynini.accep(constarined_phrase, token_type=self.symbol_table)
            A_con = self.match_all_wildcard + A + self.match_all_wildcard
            A_cons.append(A_con)
        return A_cons
    
    def BFS_search(self, wfsa, L):
        # step 0. topological sort and remove epsilon
        wfsa = wfsa.topsort()
        wfsa = wfsa.rmepsilon()

        # step 1. initialize variables
        path = []
        path_states = set() 
        dist_set = defaultdict(lambda: defaultdict(lambda: float('inf')))
        pi_set = defaultdict(lambda: defaultdict(int))
        dist_set[0][0] = 0.0

        N = wfsa.num_states()
        fstate = None
        success_flag = True
        l, final_L, L_run, L_upper = 0, None, min(L+10, int(L*1.5)), L_run*2
        valid_L_list = []
        weighted_dist = None

        # step 2. BFS explore
        while True:
            if N == 0:
                success_flag = False
                break
            
            # step 2a. explore this level
            keep_top_n = 20
            states_to_explore_at_this_level = list([state for state in dict(sorted(dist_set[l].items(), key=lambda x: x[1])[:keep_top_n]).keys() if state != fstate]) # do not explore final state
            ii = 0
            while ii < len(states_to_explore_at_this_level):
                cur_s = states_to_explore_at_this_level[ii]
                for out_edge in wfsa.arcs(cur_s):
                    e_weight, nstate, e_label = float(out_edge.weight), out_edge.nextstate, out_edge.olabel
                    if fstate is None and wfsa.final(nstate).to_string() == "0": #TODO: multiple final states exist???
                        fstate = nstate
                    new_weight = dist_set[l][cur_s] + e_weight 
                    if e_label == 0: # epsilon transition #TODO
                        if nstate not in dist_set[l] or new_weight < dist_set[l][nstate]:
                            dist_set[l][nstate] = new_weight
                            pi_set[l][nstate] = (cur_s, e_label)
                            if nstate == fstate:
                                if dist_set[l][fstate] < 1e5:
                                    valid_L_list.append((l, dist_set[l][fstate]))
                            else:
                                states_to_explore_at_this_level.append(nstate)
                                # print(f"After appending state {nstate}: {states_to_explore_at_this_level}")
                            # input("Breakpoint")
                    else:
                        if new_weight < dist_set[l+1][nstate]:
                            dist_set[l+1][nstate] = new_weight
                            pi_set[l+1][nstate] = (cur_s, e_label)
                            if nstate == fstate:
                                if dist_set[l+1][fstate] < 1e5:
                                    valid_L_list.append((l+1, dist_set[l+1][fstate]))
                    
                ii += 1

            # step 2b. increment level
            l += 1
            # step 2c. check if a valid path is obtained and rerank
            if ((l >= L_run) and (fstate is not None) and (len(valid_L_list) !=0)) or (l > L_upper):
                # t0 = time.time()
                if l > L_upper and len(valid_L_list) == 0: # abort if l > L_upper 
                    success_flag = False
                    self.length_constraint_fail_cnt += 1
                    print("Length Constraint Failed. Count:", self.length_constraint_fail_cnt)
                    break 
                if self.len_constraint_ranking == 'exact_length':
                    for i in range(len(valid_L_list)-1, 0, -1):
                        ll, dist = valid_L_list[i]
                        if ll >= L: # Prioritize long sequence
                            final_L, weighted_dist = ll, dist
                        elif final_L is None: # Only generate shorter sequence when longer sequence is not available.
                            final_L, weighted_dist = ll, dist
                            break
                        else:
                            break
                elif self.len_constraint_ranking == 'min_norm_dist':
                    final_L, weighted_dist = min([(length, llk/length) for (length, llk) in valid_L_list], key=lambda x: x[1])
                elif self.len_constraint_ranking == 'min_norm_dist_with_penalty':
                    def _compute_dist_with_penalty(weight, designated_l, real_l):
                        A = 10.0
                        factor = 1.0
                        if real_l >= designated_l:
                            factor = 1.0
                        else:
                            factor = A * np.exp(designated_l/real_l)
                        return factor * (weight/real_l)
                    reranked_list = sorted([(length, _compute_dist_with_penalty(llk, L, length)) for (length, llk) in valid_L_list], key=lambda x: x[1])
                    final_L, weighted_dist = min(reranked_list, key=lambda x: x[1])
                else:
                    raise NotImplementedError(f"Candidate ranking method {self.len_constraint_ranking} is not implemented!")
                break

            # step 3. decoding
            decoded_tokens = []
            decoded_path = []
            decoded_score = False
            decoded_str = ""
            if success_flag and (fstate in pi_set[final_L]):
                cur_node = fstate
                l = final_L
                while l > 0:
                    decoded_path.append(pi_set[l][cur_node][0])
                    token = self.symbol_table.find(pi_set[l][cur_node][1])
                    if token != '<epsilon>': # if epsilon, stay at the same level. keep exploring
                        decoded_tokens.append(token)
                        l -= 1
                    cur_node = decoded_path[-1]

                decoded_path.reverse()
                decoded_tokens.reverse()
                decoded_str = "".join(decoded_tokens)
                decoded_score = -weighted_dist
            else: # failed. Fall back to shortest path
                print("Fall back to shortest path")
                A_res = pynini.shortestpath(wfsa)
                decoded_tokens = [""]
                try:
                    decoded_tokens = A_res.string(token_type=self.symbol_table).split(' ')
                    decoded_score = -float(pynini.shortestdistance(A_res)[0].to_string()) #NOTE: negative sign = convert back to log-likelihoood
                    decoded_str = "".join(decoded_tokens)
                except:
                    decoded_score = -100.0

            # input("Breakpoint")
            logger.debug(f"L={final_L}, N={N}, decoded='{decoded_str}', score={decoded_score}, success_flag={success_flag}")
            decoded_token_ids = self.tokenizer.convert_tokens_to_ids(decoded_tokens)
            decoded_string = self.tokenizer.convert_tokens_to_string(decoded_tokens)

            return {
                'decoded_token_ids': decoded_token_ids,
                'decoded_string': decoded_string,
                'decoded_score': decoded_score
            }
    
    def topological_search(self, wfsa, L):
        # step 0. topological sort and remove epsilon
        wfsa = wfsa.topsort()
        wfsa = wfsa.rmepsilon()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # step 1. initialize variables
        # L_upper = min(L+5, int(L*1.5))
        L_upper = L+5
        fstates = [] # keep track of final states
        N = wfsa.num_states()
        f = torch.ones(size=(N, L_upper+1)).to(device) * float('inf') # f[s, l-1] records the shortest path weight from the start state to state s with exactly l edges
        f[0, 0] = 0 # initial condition. 0 cost for length 0 at start state 0
        pi = torch.zeros(size=(N, L_upper+1, 2), dtype=torch.int64).to(f.device) # pi[:,:,0]=parent pointer; pi[:,:,1]=token_id

        # step 2. do search in topological order
        for cstate in wfsa.states(): # iterate in topological order
            if wfsa.final(cstate).to_string() == "0":
                fstates.append(cstate)
            shortest_out_edge = defaultdict() # keyed by nstate
            for out_edge in wfsa.arcs(cstate): # find the shortest edge to every state
                nstate, e_weight = out_edge.nextstate, float(out_edge.weight)
                shortest_out_edge[nstate] = out_edge if nstate not in shortest_out_edge or e_weight < float(shortest_out_edge[nstate].weight) else shortest_out_edge[nstate]
            
            for nstate in shortest_out_edge:
                out_edge = shortest_out_edge[nstate]
                e_weight, nstate, e_label = float(out_edge.weight), out_edge.nextstate, out_edge.olabel
                next_f = f[cstate, :-1] + e_weight
                min_vals, min_idx = torch.min(torch.vstack([f[nstate, 1:], next_f]), dim=0)
                f[nstate, 1:] = min_vals
                pi[nstate, 1:, 0] = torch.where(min_idx.to(device, dtype=torch.bool), torch.ones_like(pi[nstate, 1:, 0], dtype=torch.int64).to(device)*cstate, pi[nstate, 1:, 0])
                pi[nstate, 1:, 1] = torch.where(min_idx.to(device, dtype=torch.bool), torch.ones_like(pi[nstate, 1:, 1], dtype=torch.int64).to(device)*e_label, pi[nstate, 1:, 1])
                # print(pi[nstate])
                # input()
            #     print(f"f[{nstate},:] = {f[nstate, :]}")
            #     print(f"pi[{nstate},:] = {pi[nstate, :]}")
            #     input(f"breakpoint - cstate={cstate} x nstate={nstate}")
            # input(f'breakpoint - cstate={cstate}')
        
        # step 3. candidate reranking
        final_f = f[fstates, :]
        min_value, min_idx = None, None
        if self.len_constraint_ranking == 'exact_length':
            min_value, min_idx = torch.min(f[fstates, L], dim=0)
            final_L, end_state = L, min_idx
        elif self.len_constraint_ranking == 'min_norm_dist':
            min_values, min_lengths = torch.min(f[fstates, :], dim=1)
            min_value, min_idx = min_values.min(dim=0)
            final_L, end_state = min_lengths[min_idx], fstates[min_idx]
        elif self.len_constraint_ranking == 'min_norm_dist_with_penalty':
            A = 10.0
            len_mat = torch.arange(L_upper+1).repeat((len(fstates), 1)).to(f.device)
            factors = torch.clip(torch.exp(A*(L/len_mat-1)), min=1.0).to(f.device) # where length penalty is implemented
            # print(f"N={N}, L={L}, fstates={fstates}")
            # print(f"factors.shape={factors.shape}, final_f.shape={final_f.shape}")
            # input()
            reweighted_f = final_f * factors
            min_values, min_lengths = torch.min(reweighted_f, dim=1)
            min_value, min_idx = min_values.min(dim=0)
            final_L, end_state = min_lengths[min_idx], fstates[min_idx]

        weighted_dist = min_value 
        # print(f"final_L={final_L}, end_state={end_state}, weighted_dist={weighted_dist}")
        # input()

        # step 4. decoding
        def _backtrace_decode(pi, final_L, fstate): # pi is the parent pointer matrix
            decoded_path = []
            decoded_tokens = []
            cur_node = fstate
            l = final_L
            while l > 0:
                decoded_path.append(pi[cur_node, l, 0])
                token = self.symbol_table.find(pi[cur_node, l, 1])
                # print(f"l={l}, decoded_path: {decoded_path}, tokens={token}, cur_node={cur_node}")
                # print(pi[cur_node, :, :])
                # input()
                decoded_tokens.append(token)
                cur_node = decoded_path[-1]
                l -= 1
            
            decoded_path.reverse()
            decoded_tokens.reverse()

            return decoded_path, decoded_tokens
        
        decoded_path, decoded_tokens = _backtrace_decode(pi, L, end_state)

        # step 5. return results
        decoded_token_ids = self.tokenizer.convert_tokens_to_ids(decoded_tokens)
        decoded_string = self.tokenizer.convert_tokens_to_string(decoded_tokens)
        # print("path:", decoded_path)
        # print(decoded_string)
        # input()

        return {
            'decoded_token_ids': decoded_token_ids,
            'decoded_string': decoded_string,
            'decoded_score': weighted_dist,
        }
    

    def dfs_memo(self, wfsa, L):
        wfsa = wfsa.rmepsilon()
        wfsa = wfsa.topsort()

        L_upper = min(L+5, int(L*1.5))
        N = wfsa.num_states()
        memo = defaultdict(lambda: defaultdict(lambda: ([], [], float('inf')))) # keyed by node, item = {L: (path, weight)}
        final_states = set()

        def _dfs_explore(cstate, cur_len):
            if len(final_states) and cur_len > L_upper: # abort when length is exceeded
                return {} # return an empty dictionary, meaning there is no valid suffix for the node (but this will then be doubly explored?)
            if wfsa.final(cstate).to_string() == "0":
                final_states.add(cstate)
                return {0: ([], [], 0.0)}
                # if self.vtrie.is_word: # must end with a valid word 
                #     return {0: ([], [], 0.0)}
                # else:
                #     return {}

            # use memo if available
            suffix_paths_dict = defaultdict(lambda: ([], [], float('inf'))) # indexed by suffix length l

            if cstate in memo: # found in memo
                suffix_paths_dict = memo[cstate]
                return suffix_paths_dict
                # if self.apply_vocab_constraint == 'trie': # work in progress
                #     suffix_paths_dict_to_return = {}
                #     for l, (suffix_path, suffix_str, suffix_weight) in suffix_paths_dict.items():
                #         # print(suffix_str)
                #         tokens_advanced = []
                #         is_valid = True
                #         for token in suffix_str:
                #             if self.vtrie.check_advance(token):
                #                 self.vtrie.advance(token)
                #                 tokens_advanced.append(token)
                #             else:
                #                 is_valid = False
                #                 break
                #         # print(self.vtrie)
                #         is_valid = self.vtrie.is_word()
                #         for token in tokens_advanced:
                #             self.vtrie.pop()
                #         if is_valid:
                #             suffix_paths_dict_to_return[l] = suffix_paths_dict[l]
                #         # print("is_valid:", is_valid)
                #         # input("(BREAKPOINT)")
                #     return suffix_paths_dict_to_return
                # else:
            else: # dfs explore
                shortest_out_edge = {}
                for out_edge in wfsa.arcs(cstate):
                    nstate, nweight, nlabel = out_edge.nextstate, float(out_edge.weight), self.symbol_table.find(out_edge.olabel)
                    # if self.apply_vocab_constraint == 'trie':
                    #     if not self.vtrie.check_advance(nlabel): continue # pruned if this is not a valid subword to follow the current vtrie state
                    shortest_out_edge[nstate] = (nstate, nlabel, nweight) if nstate not in shortest_out_edge or nweight < shortest_out_edge[nstate][-1] else shortest_out_edge[nstate]
                ordered_edges_to_explore = sorted(shortest_out_edge.items(), key=lambda x: x[1][-1])
                
                # pruning
                # prune by cumulative probability (<CUMSUM_VAL)
                CUMSUM_VAL = self.len_prune_cumprob 
                keep_top_k = 3 
                edge_probs = np.exp([-weight for _, (__, ___, weight) in ordered_edges_to_explore])
                edge_probs = edge_probs / np.sum(edge_probs)
                edge_cumsum = np.cumsum(edge_probs)

                # print("Edge cumsum", edge_cumsum)
                # prune with cap
                top_n = min(len(edge_cumsum[edge_cumsum<CUMSUM_VAL])+1, keep_top_k)
                # print("(BEFORE PRUNE) out_edges_to_explore", ordered_edges_to_explore)
                
                ordered_edges_to_explore = ordered_edges_to_explore[:top_n]

                # print(f"(AFTER PRUNE, keep={top_n}) out_edges_to_explore", ordered_edges_to_explore)
                # input("(BREAKPOINT)")
                # print(shortest_out_edge)
                # print(ordered_edges_to_explore)
                # input("(BREAKPOINT)")
                for _, (nstate, nlabel, nweight) in ordered_edges_to_explore: # greedy DFS
                    # print(f"nlabel={nlabel}")
                    # print("BEFORE ADVANCE")
                    # print(self.vtrie.active_path)
                    # if self.apply_vocab_constraint == 'trie': self.vtrie.advance(nlabel)
                    # print("AFTER ADVANCE")
                    # print(self.vtrie.active_path)
                    nstate_suffix_paths = _dfs_explore(nstate, cur_len+1)
                    # if self.apply_vocab_constraint == 'trie': self.vtrie.pop()
                    # print("BEFORE POP")
                    # print(self.vtrie.active_path)
                    # print("AFTER POP")
                    # print(self.vtrie.active_path)
                    # input("(BREAKPOINT)")
                        
                    for l, (suffix_path, suffix_str, suffix_weight) in nstate_suffix_paths.items():
                        suffix_paths_dict[l+1] = ([nstate]+suffix_path, [nlabel]+suffix_str, nweight+suffix_weight) if nweight+suffix_weight < suffix_paths_dict[l+1][-1] else suffix_paths_dict[l+1]
                memo[cstate] = suffix_paths_dict
            
            return suffix_paths_dict
        
        if self.apply_vocab_constraint == 'trie': self.vtrie.reset() # reset vtrie state before start of algorithm
        all_paths_from_start = _dfs_explore(0, 0)
        
        # if self.apply_vocab_constraint == 'trie':
        #     valid_paths = {}
        #     for l, (path, tokens, score) in all_paths_from_start.items():
        #         if self.vtrie.check_sentence(tokens):
        #             valid_paths[l] = all_paths_from_start[l]
        #         # else:
        #         #     print(tokens)
        #         #     print(self.vtrie)
        #         #     input("(BREAKPOINT)")
        #     all_paths_from_start = valid_paths

        # breakpoint()
        if len(all_paths_from_start):
            if self.len_constraint_ranking == 'exact_length':
                best_path = None
                for l_to_take in list(sorted(all_paths_from_start.keys(), reverse=True)):
                    if l_to_take < L and best_path is not None: 
                        break
                    best_path = (l_to_take, all_paths_from_start[l_to_take]) if l_to_take in all_paths_from_start else best_path
                if best_path is None:
                    print(L_upper, L, all_paths_from_start.keys())
                    # input("BREAKPOINT")
            elif self.len_constraint_ranking == 'min_dist':
                best_path = min(all_paths_from_start.items(), key=lambda x: x[1][-1])
            elif self.len_constraint_ranking == 'min_norm_dist':
                reweighted_paths = {l: (p, s, v/l) for l, (p, s, v) in all_paths_from_start.items() if l!=0}
                best_path = min(reweighted_paths.items(), key=lambda x: x[1][-1])
            elif self.len_constraint_ranking == 'min_norm_dist_with_penalty':
                A = self.len_strictness_A
                def _compute_score_with_penalty(score, cur_len):
                    factor = math.exp(A*(L/cur_len - 1)) if cur_len < L else 1.0
                    # factor = math.exp(A*(L/cur_len-1)) if cur_len < L else math.exp(A*(cur_len/L-1))
                    return factor*score
                reweighted_paths = {l: (p, s, _compute_score_with_penalty(v, l)) for l, (p, s, v) in all_paths_from_start.items() if l!=0}
                # print("all_paths", all_paths_from_start)
                # print("reweighted_paths", reweighted_paths)
                best_path = min(reweighted_paths.items(), key=lambda x: x[1][-1])
        else: # fall back to shortest path
            print("Fall back to shortest path")
            res = self.wfsa_shortest_path(wfsa)
            decoded_tokens, decoded_string, decoded_score, decoded_path = res['decoded_tokens'], res['decoded_string'], res['decoded_score'], res['decoded_path']
            # print(res)
            # DEBUG. Make sure neologism is zero
            decoded_tokens = ["▁"]
            decoded_string = "" 
            # input(f"(BREAKPOINT) WFSA shortest path decoding result: {res}")
            # A_res = pynini.shortestpath(wfsa)
            # decoded_tokens = A_res.string(token_type=self.symbol_table).split(' ')
            # decoded_score = -float(pynini.shortestdistance(A_res)[0].to_string())
            best_path = (len(decoded_tokens), ([], decoded_tokens, decoded_score))
            
        (final_L, (path, decoded_tokens, reweighted_score)) = best_path
        decoded_score = all_paths_from_start[final_L][-1] if len(all_paths_from_start) else reweighted_score
        decoded_string = self.tokenizer.convert_tokens_to_string(decoded_tokens)
        # print(f"final_L={final_L}, Constrained L={L}")
        # print("decoded string:", decoded_string)
        # print("decoded_score:", decoded_score)
        # print("reweighted score:", reweighted_score)
        # input("(BREAKPOINT)")

        return {
            'decoded_string': decoded_string,
            'decoded_score': decoded_score,
            'decoded_tokens': decoded_tokens,
            'decoded_path': path,
            'detail_info': {
                'len_reweighted_decoded_score': reweighted_score,
                'len_decode_success_flag': len(all_paths_from_start) > 0, 
            }
        }
    
    def wfsa_shortest_path(self, wfsa):
        A_res = pynini.shortestpath(wfsa)
        decoded_tokens = [""]
        try:
            decoded_tokens = A_res.string(token_type=self.symbol_table).split(' ')
            decoded_score = -float(pynini.shortestdistance(A_res)[0].to_string()) #NOTE: negative sign = convert back to log-likelihoood
        except:
            decoded_score = -100.0
            logging.warning("FAILED! NOTHING IS GENERATED!")
        decoded_str = self.tokenizer.convert_tokens_to_string(decoded_tokens)
        return {
            'decoded_tokens': decoded_tokens,
            'decoded_string': decoded_str,
            'decoded_score': decoded_score,
            'decoded_path': [],
        }
    
    def run_search(
        self,
        node_word_logits,
        links,
        graph_lens,
        forced_token_ids=None,
        specified_length_constraint=None,
    ):
        normalized_word_logits = node_word_logits.log_softmax(dim=-1)
        normalized_links = links.log_softmax(dim=-1).nan_to_num(0.0)
        normalized_node_token_logits, node_token_idx, normalized_links = self._prune_dag(
            normalized_word_logits, 
            normalized_links, 
            forced_token_ids=forced_token_ids,
            top_k=self.top_k_emissions, 
            add_forced_tokens=self.add_forced_tokens_to_dag, 
            top_k_links=self.top_k_transitions,
            dag_vocab_constraint=self.dag_vocab_constraint)
        # stime = time.time()
        batch_edges = self.make_edges(normalized_node_token_logits, node_token_idx, normalized_links, graph_lengths=graph_lens) # most time consuming step
        # print(f"{time.time()-stime} seconds to make edges")
        # decode one by one

        output_tokens = []
        output_scores = []
        output_strings = []
        decoded_paths = []
        detail_infos = []
        for batch_idx, edges in enumerate(batch_edges):
            detail_info_dict = {}

            final_state_idx = node_word_logits.size(1)
            wfsa = self.make_wfsa_from_edges(edges, final_state=final_state_idx-1)
            raw_wfsa = wfsa.copy()
            detail_info_dict.update({
                'raw_wfsa_size': raw_wfsa.num_states(),
            })

            vtrie_added_words = []

            if self.apply_vocab_constraint == 'fsa':
                if forced_token_ids is not None and len(forced_token_ids[batch_idx][0]) and self.add_vocab_dynamically: 
                    
                    constrained_phrase_tokens = [self.tokenizer.convert_ids_to_tokens(forced_phrase_ids) for forced_phrase_ids in forced_token_ids[batch_idx]]
                    A_sv, unclosed_A_sv = self._build_allowed_vocab_fsa(constrained_phrase_tokens) # acceptor for the non-categorical slot values

                    A_vocab = pynini.union(self.unclosed_all_allowed_vocab_acceptor, unclosed_A_sv).closure()
                else:
                    A_vocab = self.all_allowed_vocab_acceptor
                wfsa = wfsa @ A_vocab
                detail_info_dict.update({
                    'wfsa_size_after_fsa_vocab_constraint': wfsa.num_states(),
                })
                if wfsa.num_states() == 0:
                    wfsa = raw_wfsa

            elif self.apply_vocab_constraint == 'trie' and len(forced_token_ids[batch_idx][0]):
                constrained_phrase_tokens = [self.tokenizer.convert_ids_to_tokens(forced_phrase_ids) for forced_phrase_ids in forced_token_ids[batch_idx]]
                # input(f"(BREAKPOINT)constrained phrase tokens: {constrained_phrase_tokens}")
                for phrase in constrained_phrase_tokens: # add in constrained tokens
                    word_added = self.vtrie.add_word(phrase)
                    if word_added:
                        vtrie_added_words.append(phrase)

            if wfsa.num_states() and self.use_constraints and forced_token_ids is not None and len(forced_token_ids[batch_idx][0]):
                wfsa = wfsa.rmepsilon()
                A_cons = self.make_constrained_FSAs(forced_token_ids[batch_idx])
                for A_con in A_cons:
                    wfsa = wfsa @ A_con # intersection operation
                detail_info_dict.update({
                    'wfsa_size_after_sv_constraints': wfsa.num_states(),
                })

            decoded_tokens = ""
            decoded_score = None

            if wfsa.num_states() == 0:
                logger.warning("Got empty wfsa! Revert to raw wfsa.")
                wfsa = raw_wfsa
                detail_info_dict.update({
                    'wfsa_decoding_fail_flag': True,
                })
            
            detail_info_dict.update({
                'wfsa_size_before_decoding': wfsa.num_states(),
            })

            if specified_length_constraint is None: # shortest path decoding
                    # raw_res = self.wfsa_shortest_path(raw_wfsa)
                    # input(f"(BREAKPOINT) Got empty wfsa. Shortest path on unconstrained wfsa: {raw_res}")
                res = self.wfsa_shortest_path(wfsa)
                decoded_tokens, decoded_string, decoded_score, decoded_path = res['decoded_tokens'], res['decoded_string'], res['decoded_score'], res['decoded_path']
                # A_res = pynini.shortestpath(wfsa)
                # decoded_tokens = ""
                # try:
                #     decoded_tokens = A_res.string(token_type=self.symbol_table).split(' ')
                #     decoded_score = -float(pynini.shortestdistance(A_res)[0].to_string()) #NOTE: negative sign = convert back to log-likelihoood
                # except:
                #     logging.warning("FAILED! NOTHING IS GENERATED!")
                #     output_tokens.append([])
                #     output_scores.append(-100.0)
                #     output_strings.append(decoded_tokens)            
                #     decoded_paths.append([]) # after merging, states may have been rearranged.
                #     break
                output_tokens.append(decoded_tokens)
                output_scores.append(decoded_score)
                output_strings.append(decoded_string)
                decoded_paths.append([])
            else: # length constrained decoding done iteratively
                if self.len_search_algo == 'bfs':
                # step 0. topsort wfsa
                    res = self.BFS_search(wfsa, specified_length_constraint[batch_idx])
                    decoded_token_ids, decoded_string, decoded_score = res['decoded_token_ids'], res['decoded_string'], res['decoded_score']
                    output_tokens.append(decoded_token_ids)
                    output_scores.append(decoded_score)
                    output_strings.append(decoded_string)            
                    decoded_paths.append([]) # after merging, states may have been rearranged.
                elif self.len_search_algo == 'top_search':
                    res = self.topological_search(wfsa, specified_length_constraint[batch_idx])
                    decoded_token_ids, decoded_string, decoded_score = res['decoded_token_ids'], res['decoded_string'], res['decoded_score']
                    output_tokens.append(decoded_token_ids)
                    output_scores.append(decoded_score)
                    output_strings.append(decoded_string)            
                    decoded_paths.append([]) # after merging, states may have been rearranged.
                elif self.len_search_algo == 'dfs_memo':
                    res = self.dfs_memo(wfsa, specified_length_constraint[batch_idx])
                    decoded_string, decoded_score, decoded_path = res['decoded_string'], res['decoded_score'], res['decoded_path']
                    output_tokens.append([])
                    output_scores.append(decoded_score)
                    output_strings.append(decoded_string)
                    decoded_paths.append(decoded_path) # after merging, states may have been rearranged.
                    detail_info_dict.update(res.get('detail_info', {})) # log detail info
                else:
                    raise NotImplementedError(f"{self.len_search_algo} is not implemented. Set specified_length_constraint to None")
            
            if self.apply_vocab_constraint == 'trie':
                # input(f"(BREAKPOINT) word to remove: {vtrie_added_words}")
                for word_to_remove in vtrie_added_words:
                    self.vtrie.remove_word(word_to_remove)

            if self.eval_neologism_on_the_fly:
                tmp_eval_recorder = EvalRecorder(name='tmp', base_dir='base') 
                tmp_eval_recorder.log_sample_dict_batch({
                    'prediction': output_strings,
                    'output_token': output_tokens,
                    'output_score': output_scores
                })
                neo_eval_recorder = self.compute_neologism_op(tmp_eval_recorder)
                neo_log = neo_eval_recorder.get_sample_logs()
                for idx, pred, has_neo_word, neo_word in zip(neo_log['index'], neo_log['prediction'], neo_log['has_neo_word'], neo_log['neo_words']):
                    if has_neo_word == True:
                        print(f"pred={pred}\nneo_words={neo_word}")
                        input("(BREAKPOINT)")

            detail_infos.append(detail_info_dict)

        logger.debug(f"output_strings: {output_strings}")

        return {
            'output_tokens': output_tokens,
            'output_scores': output_scores,
            'decoded_paths': decoded_paths,
            'output_strings': output_strings,
            'detail_infos': detail_infos,
        }
                

