import torch
import pynini
from transformers import T5TokenizerFast
from runway_for_ml.utils.util import get_tokenizer
import time
import json
import pathlib
import os
import numpy as np
from collections import defaultdict
import copy

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _prune_dag(node_word_logits, links, forced_token_ids=None, top_k=3):
    """Note: only works for batch_size=1 because the way forced_token_ids is set

    Args:
        node_word_logits (_type_): _description_
        links (_type_): _description_
        forced_token_ids (_type_): _description_
        top_k (int, optional): _description_. Defaults to 3.
    """
    # batch_forced_token_ids = []
    all_forced_token_ids = []
    if forced_token_ids is not None and len(forced_token_ids):
        for t_ids in forced_token_ids[0]:
            all_forced_token_ids.extend(t_ids)

    _, top_idx = torch.topk(node_word_logits, top_k, dim=-1)

    batch_size, graph_size, _ = node_word_logits.shape

    select_idx = torch.zeros((batch_size, graph_size, top_k+len(all_forced_token_ids)), dtype=torch.long).to(node_word_logits.device)

    select_idx[:,:,:top_k] = top_idx
    if len(all_forced_token_ids):
        select_idx[:,:,top_k:] = torch.tensor(all_forced_token_ids, dtype=torch.long).to(select_idx.device)

    pruned_node_word_logits = node_word_logits.gather(-1, select_idx)
    node_token_ids = select_idx + 1 # to make up for the off-by-one due to epsilon

    return pruned_node_word_logits, node_token_ids, links

class FSTDecoder:
    def __init__(self, tokenizer_config, top_k_transitions=2, top_k_emissions=3, use_constraints=True, apply_vocab_constraint=False, vocab_file=None, use_cache_fst_file=True, add_vocab_dynamically=True, word_insertion_penalty=0.0, len_constraint_ranking='min_norm_dist_with_penalty'):
        self._build_symbol_table(tokenizer_config) # build self.tokenizer and self.symbol_table
        self.match_all_wildcard = self._build_match_all_wildcard()
        self.top_k_transitions = top_k_transitions
        self.top_k_emissions = top_k_emissions
        self.use_constraints = use_constraints
        self.add_vocab_dynamically = add_vocab_dynamically
        self.word_insertion_penalty = word_insertion_penalty

        self.apply_vocab_constraint = apply_vocab_constraint
        self.vocab_file = vocab_file
        self.use_cache_fst_file = use_cache_fst_file

        self.length_constraint_fail_cnt = 0
        self.len_constraint_ranking = len_constraint_ranking

        logger.debug(f"Word insertion penalty = {word_insertion_penalty}")
        logger.debug(f"Use constaints: {self.use_constraints}")
        logger.debug(f"Apply vocabulary constraint: {self.apply_vocab_constraint}")
        logger.debug(f"Adding vocabulary dynamically: {self.add_vocab_dynamically}")
        logger.debug(f"Length Constraint ranking method: {self.len_constraint_ranking}")
        print(f"Length Constraint ranking method: {self.len_constraint_ranking}")

        if self.apply_vocab_constraint: # enforce all vocabulary (i.e., 1-gram) are contained in the vocabulary file specified.
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
                allowed_vocab_tokens.extend(numeric_vocab)
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
            
            edges = torch.vstack([start_node_col[None:], label_col[None:], -weights_col[None:]+self.word_insertion_penalty, dest_node_col[None:]]).T
            # print("Shape before filter", edges.shape)
            # print("All start nodes are smaller than end nodes:", torch.all(edges[:, 0] < edges[:, 3]) )
            rows_to_drop = torch.logical_and(edges[:,0] >= edges[:,3], edges[:,2] > 1e5) # drop cyclic transitions and weights to large edges 
            # print("Number of rows to drop", torch.sum(rows_to_drop))
            # print(edges[rows_to_drop])
            edges = edges[~rows_to_drop]
            # print("Shape after filter", edges.shape)
            # input("breakpoint")
            batched_edges.append(edges)

        return batched_edges #NOTE: only works for batch_size = 1
    
    def make_wfsa_from_edges(self, edges):
        wfsa = pynini.accep("", token_type=self.symbol_table)
        end_state = edges[-1][0] + 1
        # while wfsa.add_state() != end_state: continue # looks stupid to me... surely there should be alternative?
        wfsa.add_states(end_state+1)
        reachable_states = set({0})
        for edge in edges.tolist():
            start_state, label_symbol_id, weight, dest_state = edge
            start_state, label_symbol_id, dest_state = int(start_state), int(label_symbol_id), int(dest_state)
            reachable_states.add(dest_state)
            if start_state in reachable_states: # because the edges are in topological order, any state not in the state are not reachable, and hence can be reduced.
                new_arc = pynini.Arc(label_symbol_id, label_symbol_id, weight, dest_state)
                wfsa.add_arc(start_state, new_arc)
        wfsa.set_start(0)
        wfsa.set_final(0, float('inf'))
        wfsa.set_final(end_state, 0.0)
        return wfsa

    def make_constrained_FSAs(self, forced_token_ids):
        A_cons = []
        for phrase_token_ids in forced_token_ids:
            constarined_phrase = " ".join(self.tokenizer.convert_ids_to_tokens(phrase_token_ids))
            A = pynini.accep(constarined_phrase, token_type=self.symbol_table)
            A_con = self.match_all_wildcard + A + self.match_all_wildcard
            A_cons.append(A_con)
        return A_cons
    
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
        normalized_node_token_logits, node_token_idx, normalized_links = _prune_dag(normalized_word_logits, normalized_links, forced_token_ids=forced_token_ids, top_k=self.top_k_emissions)
        # stime = time.time()
        batch_edges = self.make_edges(normalized_node_token_logits, node_token_idx, normalized_links, graph_lengths=graph_lens) # most time consuming step
        # print(f"{time.time()-stime} seconds to make edges")
        # decode one by one

        output_tokens = []
        output_scores = []
        output_strings = []
        decoded_paths = []
        for batch_idx, edges in enumerate(batch_edges):
            # stime = time.time()
            # t00 = time.time()
            wfsa = self.make_wfsa_from_edges(edges)
            # wfsa = wfsa.topsort() # error: cannot topsort cyclic FST
            # input("topsort the built wfsa")
            # t01 = time.time()
            # print(f"Build wfsa from edges take {t01-t00} seconds.")
            # print(f"{time.time()-stime} seconds to build wfsa")

            # if self.apply_vocab_constraint and self.use_constraints and forced_token_ids is not None and len(forced_token_ids[batch_idx][0]):
            #     A_cons = self.make_constrained_FSAs(forced_token_ids[batch_idx])
            #     A_vocab_and_cons = pynini.union(self.all_allowed_vocab_acceptor, A_cons)
            #     wfsa = wfsa @ A_vocab_and_cons

            if self.apply_vocab_constraint:
                if forced_token_ids is not None and len(forced_token_ids[batch_idx][0]) and self.add_vocab_dynamically: 
                    # t6 = time.time()
                    
                    constrained_phrase_tokens = [self.tokenizer.convert_ids_to_tokens(forced_phrase_ids) for forced_phrase_ids in forced_token_ids[batch_idx]]
                    # print("constrained_phrase_tokens", constrained_phrase_tokens)
                    A_sv, unclosed_A_sv = self._build_allowed_vocab_fsa(constrained_phrase_tokens) # acceptor for the non-categorical slot values

                    # t6_1 = time.time()
                    # print(f"Making slot values fsa takes {t6_1-t6} seconds")
                    # t6_2 = time.time()
                    # A_sv = pynini.optimize(A_sv)
                    # t6_3 = time.time()
                    # print(f"Optimizing slot value fsa takes {t6_3-t6_2} seconds")
                    A_vocab = pynini.union(self.unclosed_all_allowed_vocab_acceptor, unclosed_A_sv).closure()
                    # A_vocab = pynini.union(self.all_allowed_vocab_acceptor, A_sv)
                    # A_vocab = pynini.optimize(A_vocab)
                    # t7 = time.time()
                    # print(f"Optimizing vocabulary fsa takes {t7-t6_1} seconds")
                else:
                    A_vocab = self.all_allowed_vocab_acceptor
                # A_vocab = self.all_allowed_vocab_acceptor
                # t8 = time.time()
                wfsa = wfsa @ A_vocab
                # t9 = time.time()
                # print(f"Intersection with vocabulary fsa takes {t9-t8} seconds")

            if self.use_constraints and forced_token_ids is not None and len(forced_token_ids[batch_idx][0]):
                # t0 = time.time()
                A_cons = self.make_constrained_FSAs(forced_token_ids[batch_idx])

                # t2 = time.time()
                # A_cons = [pynini.optimize(A_con) for A_con in A_cons] # determinize & minimize
                # t3 = time.time()

                # print(f"Optimize {len(A_cons)} constraint FSAs takes {t3-t2} seconds. Average = {(t3-t2)/len(A_cons)} seconds")
                # t4 = time.time()
                for A_con in A_cons:
                    # A_con = A_con.topsort()
                    # input("topsort the constrained language")
                    wfsa = wfsa @ A_con # intersection operation
                # t5 = time.time()
                # print(f"Intersection with constrained fsa takes {t5-t4} seconds. Average = {(t5-t4)/len(A_cons)}")

            
            

            # stime = time.time()
            # wfsa = pynini.optimize(wfsa) # determinize + minimize. See https://www.opengrm.org/twiki/bin/view/GRM/PyniniOptimizeDoc
            # t10 = time.time()
            decoded_tokens = ""
            decoded_score = None
            if specified_length_constraint is None: # shortest path decoding
                # t10 = time.time()
                A_res = pynini.shortestpath(wfsa)
                # t11 = time.time()
                # print(f"Finding shortest path takes {t11-t10} seconds")
                # print(f"{time.time()-stime} seconds to run shortest path")

                # t12 = time.time()
                decoded_tokens = ""
                try:
                    decoded_tokens = A_res.string(token_type=self.symbol_table).split(' ')
                    decoded_score = -float(pynini.shortestdistance(A_res)[0].to_string()) #NOTE: negative sign = convert back to log-likelihoood
                except:
                    logging.warning("FAILED! NOTHING IS GENERATED!")
                    output_tokens.append([])
                    output_scores.append(-100.0)
                    output_strings.append(decoded_tokens)            
                    decoded_paths.append([]) # after merging, states may have been rearranged.
                    break
            else: # length constrained decoding done iteratively
                # step 0. optimize wfsa
                # A_res = pynini.optimize(wfsa)
                # wfsa
                # print(f"Optimizing WFSA takes {t1-t0} seconds")
                # epsilon removal
                # wfsa = wfsa.rmepsilon()
                # topological sort
                wfsa = wfsa.topsort() 
                input("Topological sort wfsa after constraint")
                explore_stack = [0]
                path = []
                path_states = set() 
                def _dfs(cstate):
                    for out_edge in wfsa.arcs(cstate):
                        e_weight, nstate, e_label = float(out_edge.weight), out_edge.nextstate, out_edge.olabel
                        path.append((cstate, nstate, self.symbol_table.find(e_label), e_weight))
                        if nstate in path_states: # cycle detected
                            print(path)
                            input("Cycle detected!")
                            break
                        else:
                            # print(path)
                            # input('Breakpoint')
                            path_states.add(nstate)
                            explore_stack.append(nstate)
                            _dfs(nstate)
                            path_states.remove(cstate)
                            explore_stack.pop()
                    path.pop()

                # DFS for cycle detection
                # _dfs(0)

                # step 1. initialize level set and parent set
                dist_set = defaultdict(lambda: defaultdict(lambda: float('inf')))
                pi_set = defaultdict(lambda: defaultdict(int))
                dist_set[0][0] = 0.0
                L = specified_length_constraint[batch_idx]
                N = wfsa.num_states()
                # print("L=", L, "N=", N)
                explored_states = set()
                fstate = None
                success_flag = True

                # step 2. run the iterative algorith (with pruning)
                t0 = time.time()
                l = 0
                final_L = None
                # L_run = min(L+10, int(L*1.5))
                L_run = L
                L_upper = L_run*2
                # valid_L = None
                valid_L_list = []
                weighted_dist = None
                # for s in wfsa.states():
                #     if wfsa.final(s).to_string() == "0":
                #         fstate = s
                #         break
                # print("fstate:", fstate)

                # step 3. BFS explore
                while True:
                    # print(l, fstate)
                    if N == 0: # got empty wfsa
                        success_flag = False
                        break

                    # step 3a. explore this level
                    keep_top_n = 20
                    states_to_explore_at_this_level = list([state for state in dict(sorted(dist_set[l].items(), key=lambda x: x[1])[:keep_top_n]).keys() if state != fstate]) # do not explore final state
                    
                    ii = 0
                    while ii < len(states_to_explore_at_this_level):
                        # print("States to explore at level", l, ":", states_to_explore_at_this_level)
                        cur_s = states_to_explore_at_this_level[ii]
                    # for cur_s in dist_set[l]:
                        # if cur_s in explored_states:
                        #     continue
                        # else:
                        #     explored_states.add(cur_s)
                        for out_edge in wfsa.arcs(cur_s):
                            e_weight, nstate, e_label = float(out_edge.weight), out_edge.nextstate, out_edge.olabel
                            if fstate is None and wfsa.final(nstate).to_string() == "0": #TODO: multiple final states exist???
                                fstate = nstate
                                # print(wfsa.final(nstate))
                                # print("fstate=", fstate)

                            # print("Edge:", e_weight, nstate, e_label)
                            # if e_weight < 1e-3:
                                # print(f"l={l}, e_weight={e_weight}, e_label={self.symbol_table.find(e_label)}")
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

                    # step 3b. increment level
                    l += 1
                    # print(l)

                    # t0 = time.time()
                    # if fstate in dist_set[l] and fstate in pi_set[l]:
                    #     if dist_set[l][fstate] < 1e5:
                    #         # print(dist_set[l][fstate])
                    #         valid_L_list.append((l, dist_set[l][fstate]))
                    # t1 = time.time()
                    # print(f"Comparing dist_set[l][fstate] takes {t1-t0} secs")

                    # step 3c. check if a valid path is obtained and rerank
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
                            # print("Reranked list=", reranked_list)
                            final_L, weighted_dist = min(reranked_list, key=lambda x: x[1])
                            # print("Final L=", final_L)
                        else:
                            raise NotImplementedError(f"Candidate ranking method {self.len_constraint_ranking} is not implemented!")
                        # t1 = time.time()
                        # print(f"Rank candidate takes {t1-t0} secs")
                        break

                    # dist_set[l] = dict(sorted(dist_set[l].items(), key=lambda x: x[1])[:100]) # pruning
                    # dist_set[l] = dict(sorted(dist_set[l].items(), key=lambda x: x[1])[:20]) # pruning
                    # pruned_dist_set = defaultdict(lambda: defaultdict(lambda: float('inf')))
                    # pruned_dist_set[l] = dict(sorted(dist_set[l].items(), key=lambda x: x[1])[:20]) # pruning
                    # if fstate in dist_set[l]:
                    #     pruned_dist_set[l].update({fstate: dist_set[l][fstate]})
                    # dist_set[l] = pruned_dist_set[l]

                # t1 = time.time()
                # print(f"Iterative Length Constrained runtime: {t1-t0} seconds")
                
            
                # step 3. decoding
                decoded_tokens = []
                decoded_path = []
                decoded_score = False
                decoded_str = ""
                # print("valid_L_list:", sorted(valid_L_list))
                # print(f"pi_set[{final_L}], fstate={fstate}", sorted(pi_set[final_L]))
                # print("success_flag", success_flag)
                if success_flag and (fstate in pi_set[final_L]):
                    cur_node = fstate
                    l = final_L
                    while l > 0:
                    # for l in range(final_L, 0, -1):
                        # print("l=",l)
                        # print(fstate, f"pi_set[{l}]=", sorted(pi_set[l]))
                        # print(f"dist_set[{l}]=", sorted(dist_set[l]))
                        # print(cur_node, pi_set[l][cur_node])
                        decoded_path.append(pi_set[l][cur_node][0])
                        token = self.symbol_table.find(pi_set[l][cur_node][1])
                        if token != '<epsilon>': # if epsilon, stay at the same level. keep exploring
                            decoded_tokens.append(token)
                            l -= 1
                        # else: # DEBUG show <epsilon> in generation
                            # decoded_tokens.append(token)
                        cur_node = decoded_path[-1]

                    decoded_path.reverse()
                    decoded_tokens.reverse()
                    decoded_str = "".join(decoded_tokens)
                    # decoded_score = -dist_set[final_L][fstate]
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


                # print("Decoded path:", decoded_path)
                # print("Decoded str:", decoded_str)


            # else: # length constrained decoding using Dynamic Programming
            #     # assert type(specified_length_constraint) == int, "specified_length_constraint must be an integer"
            #     #TODO length constrained decoding
            #     # step 0. optimize wfsa
            #     # t10 = time.time()
            #     # A_res = pynini.shortestpath(wfsa)
            #     # t11 = time.time()
            #     # print(f"Finding shortest path takes {t11-t10} seconds")
            #     # # print(f"{time.time()-stime} seconds to run shortest path")

            #     # # t12 = time.time()
            #     # pynini_decoded_tokens = ""
            #     # try:
            #     #     pynini_decoded_tokens = A_res.string(token_type=self.symbol_table).split(' ')
            #     #     pynini_decoded_score = -float(pynini.shortestdistance(A_res)[0].to_string()) #NOTE: negative sign = convert back to log-likelihoood
            #     #     print()
            #     #     print("pynini shortest path yields:", "".join(pynini_decoded_tokens))
            #     #     print("pynini score:", pynini_decoded_score)
            #     # except:
            #     #     print("Shortest path failed")

            #     wfsa = pynini.optimize(wfsa) #NOTE: optimize
                
            #     # step 1. Obtain weighted adjancency matrix from wfsa (optionally after constrains)
                
            #     N = wfsa.num_states()
            #     device = "cuda:0" if torch.cuda.is_available() and N <= 50000 else "cpu"

            #     fstate = None #NOTE: the final node may not be N-1 due to previous operations
            #     L = specified_length_constraint[batch_idx]
            #     print("Constrained WFSA states:", N)
            #     # print("Length Constraint:", L)
            #     E = torch.ones((N, N)).to(device) * float('inf') # E[u,v] = the weight of the path from u -> v
            #     edge_symbol_dict = {}
            #     # print("start:", wfsa.start())
            #     # print("final:", wfsa.final())
            #     t0 = time.time()
            #     for s in wfsa.states():
            #         # print("wfsa.final(s).to_string() = ", wfsa.final(s).to_string())
            #         if wfsa.final(s).to_string() == "0":
            #             fstate = s # final state found, do not add outgoing edges
            #             # print(f"fstate={fstate}")
            #             continue
            #         # print(f"state {s} final weight:", wfsa.final(s))
            #         for edge in wfsa.arcs(s):
            #             e_weight, nstate, e_label = float(edge.weight), edge.nextstate, edge.olabel
            #             if e_weight < E[s, nstate]:
            #                 try:
            #                 # print(f"weight assignment failed: edge.weight = {edge.weight}")
            #                 # print(f"Corresponding label {self.symbol_table.find(edge.olabel)}") 
            #                     E[s, nstate] = e_weight
            #                 except:
            #                     continue
            #                 edge_symbol_dict[f'{s}-{nstate}'] = e_label
            #     t1 = time.time()
            #     print(f"Convert to weighted adjacency matrix takes {t1-t0} seconds.")

            #     # setp 2. Initialize f, where f[N, L] stands for the shortest distance of all length-L paths that end in node N-1
            #     f = torch.zeros((N, L+1)).to(device)
            #     f[1:,0] = float('inf') # only state 0 is a valid start node
            #     pi = torch.zeros_like(f, dtype=torch.int).to(device) # parent pointer matrix
            #     pi[0, :] = -1
                

            #     t2 = time.time()
            #     # step 3. Use DP to find f[N, L] (shortest path weight with length L) and pi (parent pointers for retrieving the shortest path)
            #     for l in range(1, L+1):
            #         min_values, min_indices = torch.min(f[:,l-1, None] + E, axis=0)
            #         pi[:, l] = min_indices
            #         f[: ,l] = min_values
            #     t3 = time.time()
            #     print(f"DP takes {t3-t2} seconds.")
            #     # print(f)

            #     # step 4. Obtain the shortest path from parent pointer matrix pi
            #     path = []
            #     # v = N-1 # end node index
            #     v = fstate # end node index
            #     for l in range(L, 0, -1):
            #         path.append(int(pi[v, l]))
            #         v = path[-1]
            #     path.reverse()

            #     # (debug) setep 4b. shortest path algorithm (length <= L)
            #     # min_f, min_length = torch.min(f[N-1, :], axis=0)
            #     # min_f, min_length = torch.min(f[fstate, :], axis=0)
            #     # shortest_path = []
            #     # L_min = min_length
            #     # v = fstate  # end node index
            #     # for l in range(L_min, 0, -1):
            #     #     shortest_path.append(int(pi[v, l]))
            #     #     v = shortest_path[-1]
            #     # shortest_path.reverse()

            #     # shortest_decoded_tokens = []
            #     # for u, v in zip(shortest_path[:-1], shortest_path[1:]):
            #     #     if f'{u}-{v}' not in edge_symbol_dict:
            #     #         break
            #     #     else:
            #     #         token = self.symbol_table.find(edge_symbol_dict[f'{u}-{v}'])
            #     #         if token != '<epsilon>':
            #     #             shortest_decoded_tokens.append(token)
            #     # print(f"Shortest decoded score: {min_f}")
            #     # print(f"Shortest decoding (length = {min_length}): {''.join(shortest_decoded_tokens)}")

            #     # step 5. Get the decoded string that the shortest path corresponds to
            #     decoded_tokens = []
            #     for u, v in zip(path[:-1], path[1:]):
            #         if f'{u}-{v}' not in edge_symbol_dict:
            #             break
            #         else:
            #             token = self.symbol_table.find(edge_symbol_dict[f'{u}-{v}'])
            #             if token != '<epsilon>':
            #                 decoded_tokens.append(token)
            #     # decoded_tokens = "".join(decoded_tokens)
            #     decoded_score = f[fstate, L].item()
            #     print("decoded score:", decoded_score)
            #     print(f"decoded_tokens (length = {L}):", decoded_tokens)


            # t13 = time.time()
            # print(f"Decode shortest path (.string() call) takes {t13-t12} seconds")

            decoded_token_ids = self.tokenizer.convert_tokens_to_ids(decoded_tokens)
            decoded_string = self.tokenizer.convert_tokens_to_string(decoded_tokens)
            
            output_tokens.append(decoded_token_ids)
            output_scores.append(decoded_score)
            output_strings.append(decoded_string)            
            decoded_paths.append([]) # after merging, states may have been rearranged.
            logger.debug(f"Length Constrain Failed Count={self.length_constraint_fail_cnt}")
        
        return {
            'output_tokens': output_tokens,
            'output_scores': output_scores,
            'decoded_paths': decoded_paths,
            'output_strings': output_strings,
        }
                

