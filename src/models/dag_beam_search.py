import copy
import heapq
import torch
from abc import ABC, abstractmethod
from typing import List

    
class DAGBeamHypothesis:
    def __init__(self, score=0.0, tokens=None, path=None, max_path_length=-1, constraints=[]):
        self.score = score 
        self.tokens = tokens if tokens is not None else []
        self.path = path if path is not None else []
        self.max_path_length = max_path_length
        self.constraints = constraints
    
    @classmethod
    def copy_from(cls, other):
        return cls(
            score=copy.copy(other.score),
            tokens=copy.copy(other.tokens),
            path=copy.copy(other.path),
            max_path_length=other.max_path_length,
            constraints=[cons.copy(stateful=True) for cons in other.constraints],
        )
    
    def __repr__(self) -> str:
        return str(self)
    
    def __str__(self) -> str:
        return f"tokens: {self.tokens}; path: {self.path}; score: {self.score}"
    
    def __lt__(self, other): # if beam1 < beam2, than beam1.score > beam2.score
        return -self.score < -other.score # invert score so that heapq min heap can be used. Interpreted as "loss" when ranking
    
    @property
    def cur_node(self):
        return self.path[-1]
    
    def extend(self, next_token, next_vertex, score_increment):
        self.tokens.append(next_token if isinstance(next_token, int) else next_token.item())
        self.path.append(next_vertex if isinstance(next_vertex, int) else next_vertex.item())
        self.score += score_increment.item()
        for constraint in self.constraints:
            constraint.update(self.tokens[-1])
    
    def is_done(self):
        return self.path[-1] == self.max_path_length-1
    
    def is_valid(self): # overwritten for constraiend beam search
        if not self.is_done():
            return True
        elif len(self.constraints) == 0:
            return True
        else:
            return self.completed_all_constraints()
           
    
    def completed_all_constraints(self):
        completed = True
        for constraint in self.constraints:
            completed = completed and constraint.completed
        return completed
    
    def unmet_constraints_num(self):
        ans = 0
        for constraint in self.constraints:
            ans += constraint.remaining()
        return ans


class Constraint(ABC):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    ```py
    completed = False
    while not completed:
        _, completed = constraint.update(constraint.advance())
    ```

    will always terminate (halt).
    """

    def __init__(self):
        # test for the above condition
        self.test()

    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        counter = 0
        completed = False
        while not completed:
            if counter == 1:
                self.reset()
            advance = self.advance()
            if not self.does_advance(advance):
                raise Exception(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            stepped, completed, reset = self.update(advance)
            counter += 1

            if counter > 10000:
                raise Exception("update() does not fulfill the constraint.")

        if self.remaining() != 0:
            raise Exception("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Return:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Return:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Return:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

class PhrasalConstraint(Constraint):
    r"""
    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    """

    def __init__(self, token_ids: List[int]):
        super(Constraint, self).__init__()

        # token_ids can be empty.
        # if not isinstance(token_ids, list) or len(token_ids) == 0:
            # raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        # if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
        #     raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        self.token_ids = token_ids

        self.seqlen = len(self.token_ids)
        self.lps_arr = self._make_lps_arr() # longest prefix array
        self.fulfilled_idx = -1  # the index of the currently fulfilled step
        self.completed = False if self.seqlen > 0 else True
    
    def _make_lps_arr(self):
        pre_len = 0
        M = len(self.token_ids)
        lps = [0] * M
        i = 1
        while i < M:
            if self.token_ids[i] == self.token_ids[pre_len]:
                pre_len += 1
                lps[i] = pre_len
                i += 1
            elif pre_len != 0:
                pre_len = lps[pre_len-1]
            else:
                lps[i] = 0
                i += 1
        return lps

    def advance(self):
        if self.completed:
            return None
        return self.token_ids[self.fulfilled_idx + 1]

    def does_advance(self, token_id: int):
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False

        return token_id == self.token_ids[self.fulfilled_idx + 1]
    
    def reset(self):
        self.fulfilled_idx = -1
        self.completed = False

    def update(self, token_id):
        if self.completed: # if completed already, just return completed
            return False, True, False

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id): # match
            self.fulfilled_idx += 1
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):
                completed = True
            self.completed = completed
        else: # no match
            while self.fulfilled_idx != -1:
                self.fulfilled_idx = self.lps_arr[self.fulfilled_idx] - 1
                if self.does_advance(token_id):
                    self.fulfilled_idx += 1
                    stepped = True
                    break
            if self.fulfilled_idx == -1:
                reset = True
            # failed to make progress.
        return stepped, completed, reset

    def remaining(self):
        return self.seqlen - (self.fulfilled_idx + 1)
    
    def fulfilled_step(self):
        return self.fulfilled_idx + 1
    
    def is_active(self):
        return self.fulfilled_idx != -1

    def copy(self, stateful=False):
        new_constraint = PhrasalConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed

        return new_constraint
    
    def has_completed(self):
        return self.remaining() == 0
    
    def __repr__(self) -> str:
        return f"Constraint: {self.token_ids}; Fulfilled idx: {self.fulfilled_idx}"
    

class DAGBeamSearch:
    def __init__(
        self, 
        node_word_logits, # [i, u] = Probability of node i emitting token u  
        links, # [i, j] = Probability of node i transitioning to node j
        graph_length, # length of the graph
        beam_size=5,
        bos_token_id=0,
        forced_token_ids=[],
    ):
        self.beam_size = beam_size
        self.bos_token_id = bos_token_id
        self.L = graph_length # length of graph, end index L-1

        self.node_word_logits = node_word_logits
        self.output_word_logits = node_word_logits.log_softmax(dim=-1)
        self.emit_token_logits, self.emit_tokens = self.output_word_logits.max(dim=-1)
        self.links = links.log_softmax(dim=-1).nan_to_num(0.0) # normalized probability
        self.top_k_links_logits, self.top_k_next_vertices = torch.topk(self.links, self.beam_size, dim=-1)

        self.forced_token_ids = forced_token_ids
        self.constraints = []
        for forced_word in self.forced_token_ids:
            self.constraints.append(PhrasalConstraint(forced_word))
    
    def expand_beam(self, beam, top_k=-1):
        next_vertices = self.top_k_next_vertices[beam.cur_node]
        next_links_logits = self.top_k_links_logits[beam.cur_node]

        next_tokens = self.emit_tokens[next_vertices]
        next_tokens_logits = self.emit_token_logits[next_vertices]
        all_candidates = []
        for next_vertex, next_token, next_link_logit, next_token_logit in zip(next_vertices, next_tokens, next_links_logits, next_tokens_logits):
            incremental_score = next_link_logit + next_token_logit
            candidate_beam = DAGBeamHypothesis.copy_from(beam)
            candidate_beam.extend(next_token, next_vertex, incremental_score)
            all_candidates.append(candidate_beam)
            if len(all_candidates) == top_k:
                break
        return all_candidates


    def prune_beams(self, beams):
        if len(beams) <= self.beam_size:
            return beams
        top_k_beams = heapq.nsmallest(self.beam_size, beams) #NOTE: __lt__ is implemented as loss comparison
        return top_k_beams
    
    def run_search(self):
        beam_init = DAGBeamHypothesis(score=0.0, tokens=[self.bos_token_id], path=[0], max_path_length=self.L, constraints=self.constraints)
        active_beams = [beam_init]
        finished_beams = [DAGBeamHypothesis(score=float("-inf"), tokens=[0], path=[0], max_path_length=self.L, constraints=self.constraints)] # max-heap
        while len(active_beams): # run breadth-first search (BFS)
            this_step_active_beams_size = len(active_beams)
            for i in range(this_step_active_beams_size):
                candidate_beams = self.expand_beam(active_beams[i])
                for beam in candidate_beams:
                    if beam.is_valid(): # in constrained version, a finished beam without satisfying constraints is not considered valid
                        if beam.score > finished_beams[0].score: # check if score is better than best finished score
                            if beam.is_done():
                                heapq.heappush(finished_beams, beam) # min heap in loss (-score) = max heap in score 
                            else:
                                active_beams.append(beam)
                        else: # discard beam that will never be selected
                            continue
                    else: # discard if invalid
                        continue
            active_beams = active_beams[this_step_active_beams_size:] # remove beams on this level
            active_beams = self.prune_beams(active_beams) # prune beam to keep beam_size <= k
        top_beam = finished_beams[0]
        return top_beam
    
    # def run_dfs_search(self):
    #     beam_init = DAGBeamHypothesis(score=0.0, tokens=[self.bos_token_id], path=[0], max_path_length=self.L)
    #     active_beams = [beam_init] # a min-loss heap
    #     finished_beams = [DAGBeamHypothesis(score=float("-inf"), tokens=[0], path=[0], max_path_length=self.L)] # max-heap
    #     while len(active_beams):
    #         top_beam = heapq.heappop(active_beams)
    #         candidate_beams = self.expand_beam(top_beam)
    #         for beam in candidate_beams:
    #             if beam.is_valid(): # in constrained version, a finished beam without satisfying constraints is not considered valid
    #                 if beam.score > finished_beams[0].score: # check if score is better than best finished score
    #                     if beam.is_done():
    #                         heapq.heappush(finished_beams, beam) # min heap in loss (-score) = max heap in score 
    #                     else:
    #                         heapq.heappush(active_beams, beam)
    #                 else: # discard beam that will never be selected
    #                     continue
    #             else: # discard if invalid
    #                 continue 
    #         active_beams = self.prune_beams(active_beams) # prune beam to keep beam_size <= k
    #     res_beam = finished_beams[0]
    #     return res_beam
    

class DAGBeamSearchWithConstraints(DAGBeamSearch):
    def __init__(
        self, 
        node_word_logits, # [i, u] = Probability of node i emitting token u  
        links, # [i, j] = Probability of node i transitioning to node j
        graph_length, # length of the graph
        beam_size=5,
        bos_token_id=0,
        forced_token_ids=[],
        use_dynamic_beam_size=False,
    ):
        super().__init__(
            node_word_logits=node_word_logits,
            links=links,
            graph_length=graph_length,
            beam_size=beam_size,
            bos_token_id=bos_token_id,
            forced_token_ids=forced_token_ids,
        ) 
        self.total_constraints_length = sum([len(forced_word) for forced_word in forced_token_ids]) #TODO: handle padding
        self.use_dynamic_beam_size = use_dynamic_beam_size
        
        
    def expand_beam(self, beam):
        all_candidates = []
        greedy_candidates = super().expand_beam(beam, top_k=1)
        all_candidates.extend(greedy_candidates)

        # expand 1: emit constraint tokens
        for greedy_beam in greedy_candidates:
            next_vertex = greedy_beam.path[-1]
            greedy_token = greedy_beam.tokens[-1]
            if not beam.completed_all_constraints(): # if there are still unsatisfied constraints
                for constraint in beam.constraints:
                    if constraint.has_completed() or greedy_token == constraint.advance(): # do nothing if greedy outputs the next constraint token or constraint is already finished
                        continue
                    next_token = constraint.advance()
                    next_token_logit = self.output_word_logits[next_vertex, next_token]
                    candidate_beam = DAGBeamHypothesis.copy_from(beam)
                    score_increment = next_token_logit + self.links[candidate_beam.path[-1], next_vertex]

                    candidate_beam.extend(next_token=next_token, next_vertex=next_vertex, score_increment=score_increment)
                    all_candidates.append(candidate_beam)

        return all_candidates
                
    
    def _counting_sort_unmet_and_assign_and_sort_step(self, beams):
        """Beams are assumed to be sorted by scores in decending order

        Args:
            beams (_type_): _description_
        """
        # sort by the number of unmet constraints (count of un-realized tokens)
        unmet_banks = [[] for _ in range(self.total_constraints_length+1)]
        for beam in beams:
            unmet_num = beam.unmet_constraints_num()
            cur_step = len(unmet_banks[unmet_num])
            unmet_banks[unmet_num].append((cur_step, beam)) # each item is (step, beam)

        sorted_beams = []
        for beams_in_bank in unmet_banks:
            sorted_beams.extend(beams_in_bank)

        # now sorted_beams are sorted within each bank, and the step variables are assigned.
        sorted_beams = [beam for step, beam in sorted(sorted_beams, key=lambda x: x[0])] # sort by step, in ascending order
        num_banks = len([1 for bank in unmet_banks if len(bank)]) # number of non-empty banks
        return sorted_beams, num_banks


    def prune_beams(self, beams):
        score_sorted_beams = sorted(beams, key=lambda beam: beam.score, reverse=True)
        sorted_beams, num_banks = self._counting_sort_unmet_and_assign_and_sort_step(score_sorted_beams)
        cutoff_idx = num_banks if self.use_dynamic_beam_size and num_banks > self.beam_size else self.beam_size
        return sorted_beams[:cutoff_idx]

        

if __name__ == '__main__':
    # beam1 = DAGBeamHypothesis(score=torch.tensor(0.0), tokens=torch.tensor([1,2,3]), path=torch.tensor([1,2,3]))
    # beam2 = DAGBeamHypothesis.copy_from(beam1)
    # print(beam1)
    # print(beam2)
    # beam1.extend(2, 10, 0.1)
    # print(beam1)
    # print(beam2)
    # print(heapq.nsmallest(1, [beam1, beam2]))

    cbeam1 = DAGBeamHypothesis(constraints=[PhrasalConstraint([0, 1, 2, 0, 1, 2, 3])])
    for idx, token in enumerate([0, 1, 2, 0, 1, 2, 0, 1, 2, 3]):
        cbeam1.extend(torch.tensor(token, dtype=torch.long), torch.tensor(1, dtype=torch.long), torch.tensor(0.1))
        print(idx, cbeam1.completed_all_constrains())
    pass
    