import torch
import logging
from models.dag_beam_search import DAGBeamSearch, DAGBeamSearchWithConstraints
from models.fst_decode import FSTDecoder
from collections import defaultdict
# from models.fst_decode_archive import FSTDecoder
import time

logger = logging.getLogger(__name__)

class NARGenerationMixin:
    """
    A class containing all functions for non-autoregressive generation methods. To be used as a mixin for non-autoregressive models.
    """
    

    def _prepare_model_encoder_inputs(
        self, 
        inputs, 
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        **model_kwargs,
    ):
        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [torch.int, torch.long]
        is_pad_token_in_inputs = (pad_token_id is not None) and (pad_token_id in inputs)
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or (pad_token_id != eos_token_id)
        
        attn_mask = None
        if is_input_ids and is_pad_token_in_inputs:
            attn_mask = inputs.ne(pad_token_id).long()
        else:
            raise RuntimeError('cannot create attn_mask for encoder inputs. inputs is not input_ids')
        return inputs, attn_mask
    
    def nar_greedy_decode(
        self,
        node_word_logits,
        links,
        decoder_input_ids,
        sample_path_with_joint=False,
        emission_beta=1.0,
        **model_kwargs,
    ):
        output_word_logits = node_word_logits.log_softmax(dim=-1)
        max_logits, max_logits_tokens = output_word_logits.max(dim=-1)
        graph_lens = (~decoder_input_ids.eq(self.pad)).sum(dim=-1)
        # max_prob_tokens = max_prob_tokens.to_list()
        
        normalized_links = links.log_softmax(dim=-1).nan_to_num(0.0)
        if sample_path_with_joint: # take output logits into account
            links = links + max_logits.unsqueeze(dim=1) * emission_beta
        if sample_path_with_joint: # take output logits into account
            links = links + max_logits.unsqueeze(dim=1) * emission_beta
        links_logits, links_idx = normalized_links.max(dim=-1)
        # links_idx = links.shape[-1] - links_idx
        node_nums = links.shape[1]

        paths = []
        decoded_token_ids = []
        decoded_score = []
        for i, (linkage, graph_len) in enumerate(zip(links_idx, graph_lens)):
            # this_score = torch.tensor(max_logits[i][0]).clone().detach()
            this_score = max_logits[i][0].clone().detach()
            this_path = [links_idx[i][0].item()]
            this_decoded_token_ids = [max_logits_tokens[i][0]]
            j = 0
            while j != graph_len - 1:
                j = linkage[j]
                if j == node_nums - 1:
                    break
                this_path.append(links_idx[i][j].item())
                this_decoded_token_ids.append(max_logits_tokens[i][j])
                this_score += max_logits[i][j] + links_logits[i][j]
            paths.append(this_path)
            decoded_token_ids.append(this_decoded_token_ids)
            decoded_score.append(this_score.item())
        
        output_tokens = decoded_token_ids
        output_scores = decoded_score

        return {
            'output_tokens': output_tokens,
            'output_scores': output_scores,
            'decoded_paths': paths,
        }

    def viterbi_decode(
        self,
        node_word_logits,
        links,
        decoder_input_ids,
        sample_path_with_joint=False,
        emission_beta=1.0,
        **model_kwargs,
    ): #TODO
        # Step 1: Log softmax and initialization
        output_word_logits = node_word_logits.log_softmax(dim=-1)
        graph_lens = (~decoder_input_ids.eq(self.pad)).sum(dim=-1)
        normalized_links = links.log_softmax(dim=-1).nan_to_num(0.0)
        if sample_path_with_joint:
            max_logits, _ = output_word_logits.max(dim=-1)
            normalized_links += max_logits.unsqueeze(dim=1) * emission_beta

        # Step 2: Viterbi forward pass
        batch_size, graph_length, _ = links.size()
        max_length = graph_length // 2
        viterbi_scores = torch.full((graph_length, max_length), -float('inf')).to(node_word_logits.device)
        viterbi_indices = torch.zeros_like(viterbi_scores, dtype=torch.long)
        viterbi_scores[0, 0] = 0  # Initialize start node scores to 0

        for i in range(1, max_length):
            score_with_link = viterbi_scores[:, i-1].unsqueeze(1) + normalized_links[0, :, :]
            viterbi_scores[:, i], viterbi_indices[:, i] = score_with_link.max(dim=0)

        # Step 3: Viterbi backtracking
        paths = []
        decoded_token_ids = []
        decoded_scores = []
        for batch_index, graph_len in enumerate(graph_lens):
            path = []
            last_index = graph_len - 1
            pred_length = viterbi_scores[last_index, :].max(dim=-1)[1]
            current_vertex = last_index.item()
            while pred_length >= 0:
                path.insert(0, current_vertex)
                current_vertex = viterbi_indices[current_vertex, pred_length].item()
                pred_length -= 1
            paths.append(path)

            # Decode token ids and calculate path score
            decoded_path_token_ids = [node_word_logits[batch_index, path_step].argmax().item() for path_step in path]
            decoded_token_ids.append(decoded_path_token_ids)
            path_score = sum([output_word_logits[batch_index, step, token_id] for step, token_id in zip(path, decoded_path_token_ids)])
            decoded_scores.append(path_score.item())

        # Step 4: Prepare output format
        return {
            'output_tokens': decoded_token_ids,
            'output_scores': decoded_scores,
            'decoded_paths': paths,
        }


    def dag_beam_search(
        self,
        node_word_logits,
        links,
        decoder_input_ids,
        beam_size,
        forced_token_ids=None,
        pad_token_id=0,
        use_dynamic_beam_size=False, # If True, beam size >= number of constrained banks.
        **model_kwargs,
    ):
        batch_size = node_word_logits.shape[0]
        graph_lens = (~decoder_input_ids.eq(self.pad)).sum(dim=-1)
        output_tokens = []
        output_scores = []
        decoded_paths = []
        for batch_idx in range(batch_size):
            graph_length = graph_lens[batch_idx]
            this_links = links[batch_idx, :graph_length, :graph_length]
            this_node_word_logits = node_word_logits[batch_idx, :graph_length]
            this_forced_token_ids = forced_token_ids[batch_idx] if forced_token_ids is not None else None

            dag_beamsearch = None
            if this_forced_token_ids is None:
                dag_beamsearch = DAGBeamSearch(
                    node_word_logits=this_node_word_logits,
                    links=this_links,
                    graph_length=graph_length,
                    beam_size=beam_size,
                )
            else:
                # forced_ids_length = (~forced_token_ids.eq(pad_token_id)).sum(dim=-1)
                # forced_token_ids.tolist()
                # for i in range(len(forced_token_ids)):
                #     forced_token_ids[i] = forced_token_ids[i][:forced_ids_length[i]]
                dag_beamsearch = DAGBeamSearchWithConstraints(
                    node_word_logits=this_node_word_logits,
                    links=this_links,
                    graph_length=graph_length,
                    beam_size=beam_size,
                    forced_token_ids=this_forced_token_ids,
                    use_dynamic_beam_size=use_dynamic_beam_size,
                )
            top_beam = dag_beamsearch.run_search()

            output_tokens.append(top_beam.tokens)
            output_scores.append(top_beam.score)
            decoded_paths.append(top_beam.path)

        return {
            'output_tokens': output_tokens,
            'output_scores': output_scores,
            'decoded_paths': decoded_paths,
        }
    
    def fst_decode(
        self,
        node_word_logits,
        links,
        decoder_input_ids,
        tokenizer_config,
        specified_length_constraint=None,
        top_k_transitions=2,
        top_k_emissions=3,
        forced_token_ids=None,
        use_constraints=True,
        apply_vocab_constraint=False,
        vocab_file=None,
        use_cache_fst_file=True,
        add_vocab_dynamically=True,
        word_insertion_penalty=0.0,
        **kwargs,
    ):
        graph_lens = (~decoder_input_ids.eq(self.pad)).sum(dim=-1) 
        # stime = time.time()
        if not 'fst_decoder' in self.__dict__:
            self.fst_decoder = FSTDecoder(
                tokenizer_config, 
                top_k_transitions=top_k_transitions,
                top_k_emissions=top_k_emissions,
                use_constraints=use_constraints,
                apply_vocab_constraint=apply_vocab_constraint,
                vocab_file=vocab_file,
                add_vocab_dynamically=add_vocab_dynamically,
                use_cache_fst_file=use_cache_fst_file,
                word_insertion_penalty=word_insertion_penalty,
                **kwargs
            )
        decoded_output = self.fst_decoder.run_search(node_word_logits, links, graph_lens, forced_token_ids=forced_token_ids, specified_length_constraint=specified_length_constraint)
        # print(f"{time.time() - stime} seconds to run fst decode on one batch")
        return decoded_output

    def nar_generate(
        self,
        inputs,
        max_length=None,
        min_length=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        decode_strategy=None,
        beam_size=None,
        forced_token_ids=None,
        decode_params={},
        specified_length_constraint=None,
        **model_kwargs,
    ):
        bos_token_id = bos_token_id or self.config.bos_token_id
        pad_token_id = pad_token_id or self.config.pad_token_id
        eos_token_id = eos_token_id or self.config.eos_token_id
        if pad_token_id is None and eos_token_id is not None:
            if model_kwargs.get("attention_mask", None) is None:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
            pad_token_id = eos_token_id
        
        encoder_input_ids, encoder_attn_mask = self._prepare_model_encoder_inputs(inputs, bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id)
        
        outputs = self.forward(input_ids=encoder_input_ids, attention_mask=encoder_attn_mask)
        out_node_word_logits, links, decoder_input_ids = outputs['node_word_logits'], outputs['links'], outputs['decoder_input_ids']

        decoded_output = None
        if decode_strategy == 'nar_greedy':
            decoded_output = self.nar_greedy_decode(out_node_word_logits, links, decoder_input_ids, **decode_params)
        elif decode_strategy == 'lookahead': #TODO
            decoded_output = self.nar_greedy_decode(out_node_word_logits, links, decoder_input_ids, sample_path_with_joint=True, **decode_params) 
        elif decode_strategy == 'viterbi': #TODO
            decoded_output = self.viterbi_decode(out_node_word_logits, links, decoder_input_ids, **decode_params)
        elif decode_strategy == 'joint_viterbi': #TODO
            decoded_output = self.viterbi_decode(out_node_word_logits, links, decoder_input_ids, sample_path_with_joint=True, **decode_params)
        elif decode_strategy == 'beam_search_with_lm': #TODO
            decoded_output = self.beam_search_with_lm(out_node_word_logits, links, decoder_input_ids, **decode_params)
        elif decode_strategy == 'lookahead': #TODO
            decoded_output = self.nar_greedy_decode(out_node_word_logits, links, decoder_input_ids, sample_path_with_joint=True, **decode_params) 
        elif decode_strategy == 'viterbi': #TODO
            decoded_output = self.viterbi_decode(out_node_word_logits, links, decoder_input_ids, **decode_params)
        elif decode_strategy == 'joint_viterbi': #TODO
            decoded_output = self.viterbi_decode(out_node_word_logits, links, decoder_input_ids, sample_path_with_joint=True, **decode_params)
        elif decode_strategy == 'beam_search_with_lm': #TODO
            decoded_output = self.beam_search_with_lm(out_node_word_logits, links, decoder_input_ids, **decode_params)
        elif decode_strategy == 'dag_beam_search':
            decoded_output = self.dag_beam_search(out_node_word_logits, links, decoder_input_ids, beam_size=beam_size, **decode_params)
        elif decode_strategy == 'constrained_dag_beam_search':
            decoded_output = self.dag_beam_search(out_node_word_logits, links, decoder_input_ids, beam_size=beam_size, forced_token_ids=forced_token_ids, **decode_params) # DEBUGGING: [2018, 132, 5] for "Hi there" and [100, 19] for "This is"
        elif decode_strategy == 'fst_shortest_path':
            decoded_output = self.fst_decode(out_node_word_logits, links, decoder_input_ids, forced_token_ids=forced_token_ids, specified_length_constraint=specified_length_constraint, **decode_params)
        elif decode_strategy == 'greedy+fst':
            decoded_output = self.nar_greedy_decode(out_node_word_logits, links, decoder_input_ids, **decode_params)
            wfsa_decoded_output = self.fst_decode(out_node_word_logits, links, decoder_input_ids, forced_token_ids=None, specified_length_constraint=None, **decode_params)
            detail_infos = wfsa_decoded_output.get('detail_infos', [None]*len(out_node_word_logits))
            decoded_output['detail_infos'] = detail_infos
            # detail_infos = wfsa_decoded_output.get('detail_infos', {})
            # detail_info_list = defaultdict(list)
            # for info in detail_infos:
            #     for k, v in info.items(): 
            #         detail_info_list[k].append(v)
            # decoded_output.update(detail_info_list)
        else:
            raise NotImplementedError(f"{decode_strategy} has not been implemented.") 
        
        return {
            'decoded_output': decoded_output,
            'node_word_logits': out_node_word_logits,
            'links': links,
        }
