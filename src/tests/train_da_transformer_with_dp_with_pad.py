import torch
import torch.nn.functional as F
import copy
import time

torch.manual_seed(2023) # for reproducibility


M = 5 # target token sequence length
L = 10 # number of nodes in the DAG = graph size
bs = 2 # batch size
V = 2 # size of vocabulary 
PAD_TOKEN = -1 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def _add_seq_with_prefix(prefix, target_seq_length, total_node_num, res):
    if len(prefix) == target_seq_length:
        res.append(prefix)
    else:
        last_node_idx = prefix[-1] if len(prefix) else -1
        for next_node_idx in range(last_node_idx+1, total_node_num):
            new_prefix = prefix + [next_node_idx]
            _add_seq_with_prefix(new_prefix, target_seq_length, total_node_num, res)

def get_all_seq_of_length(L, M):
    all_sequences = []
    _add_seq_with_prefix([], M-2, L-2, all_sequences)
    all_sequences = torch.tensor(all_sequences) + 1
    all_sequences = torch.hstack((torch.zeros(len(all_sequences))[:,None], all_sequences, torch.ones(len(all_sequences))[:,None]*(L-1))).long()
    return all_sequences

def brute_force_compute_likelihood(links, node_token_emit_prob, paths, labels):
    res = []
    path_likelihood_dict = [{} for _ in range(bs)]
    for b in range(bs):
        total_prob = 0
        for path in paths:
            path_log_prob = 0
            for u in range(path.shape[0]):
                v = u+1
                path_log_prob += node_token_emit_prob[b, path[u], labels[b][u]]
                path_log_prob += links[b, path[u], path[v]] if v < len(path) else 0
            total_prob += torch.exp(path_log_prob)
            path_likelihood_dict[b]['-'.join([str(node) for node in path.tolist()])] = path_log_prob
        total_log_prob = total_prob.log()
        res.append(total_log_prob)
    return res, path_likelihood_dict

def brute_force_compute_likelihood_one_batch(links, node_token_emit_prob, paths, labels, tgt_len):
    res = []
    path_likelihood_dict = {}
    total_prob = 0
    for path in paths:
        path_log_prob = 0
        for u in range(tgt_len):
            v = u+1
            path_log_prob += node_token_emit_prob[path[u], labels[u]]
            path_log_prob += links[path[u], path[v]] if v < len(path) else 0
        total_prob += torch.exp(path_log_prob)
        path_likelihood_dict['-'.join([str(node) for node in path.tolist()])] = path_log_prob
    total_log_prob = total_prob.log()
    return total_log_prob, path_likelihood_dict

def dp_compute_likelihood(links, node_token_emit_prob, tgt_lengths, labels):
    f = torch.zeros((bs, M, L), device=device).fill_(float("-inf")) # log-likelihood
    # f[f==0] = float("-inf") # log likelihood
    b_idx = torch.arange(bs)
    f[b_idx, 0, 0] = node_token_emit_prob[b_idx, 0,labels[b_idx, 0]]
    for i in torch.arange(1, M):
        for u in torch.arange(i, L):
            trans_log_prob = torch.logsumexp(f[b_idx,i-1,:u]+links[b_idx,:u, u], dim=1)
            emit_log_prob = node_token_emit_prob[b_idx,u,labels[b_idx,i]].squeeze(-1)
            f[b_idx, i, u] = emit_log_prob +trans_log_prob
    return f[b_idx, tgt_lengths-1, L-1], f

def viterbi_best_path_recursive(links, node_token_emit_prob, labels, this_node_idx, this_tar_idx, tar_len, path):
    if this_tar_idx == tar_len-1: # when end of target is reached at the final node
        if this_node_idx == L-1:
            path[this_tar_idx] = this_node_idx.long()
            return node_token_emit_prob[this_node_idx, labels[this_tar_idx]], path
        else:
            return float("-inf"), [] # invalid path
    else:
        if this_node_idx+1 >= L:
            return float("-inf"), [] # invalid path
        next_score_and_path = [viterbi_best_path_recursive(links, node_token_emit_prob, labels, next_node_idx, this_tar_idx+1, tar_len, path)
            for next_node_idx in torch.arange(start=this_node_idx.item()+1, end=L)]
        this_score_and_path = [
            (
                next_score+node_token_emit_prob[this_node_idx, labels[this_tar_idx]]+links[this_node_idx, next_path[this_tar_idx+1]], 
                next_path
            )
            for next_score, next_path in next_score_and_path if len(next_path)
        ]
        if len(this_score_and_path) == 0:
            return float("-inf"), []
        this_best_score, this_best_path = copy.deepcopy(max(this_score_and_path))
        if len(this_best_path) == 0:
            return float("-inf"), []
        this_best_path[this_tar_idx] = this_node_idx
        return this_best_score, this_best_path
    
def viterbi_best_path_batch(links, node_token_emit_prob, labels):
    tgt_lengths = (~labels.eq(PAD_TOKEN)).sum(1)
    f = torch.zeros((bs, L, M))
    best_paths = [[[[0] for _ in range(M)] for _ in range(L)] for _ in range(bs)]
    f[f==0] = float("-inf") # log likelihood
    batch_idx = torch.arange(bs)
    f[batch_idx, 0, 0] = node_token_emit_prob[batch_idx, 0, labels[batch_idx,0]]
    for j in range(1,M):
        for i in range(j,L):
            max_llk, max_indices = torch.max(f[batch_idx, :i, j-1]+links[batch_idx, :i, i], dim=1)
            f[batch_idx, i, j] = max_llk + node_token_emit_prob[batch_idx, i, labels[batch_idx, j]]
            for b_idx in range(bs):
                best_paths[b_idx][i][j] = best_paths[b_idx][max_indices[b_idx].item()][j-1] + [i]
    return f[batch_idx, L-1, tgt_lengths-1], [bp[L-1][tgt_lengths[b_i]-1] for b_i, bp in enumerate(best_paths)]


def torch_dag_logsoftmax_gather_inplace(node_word_logits, select_idx):
    r""" Fused operation of log_softmax and gather"""
    r"""Comments:
    logits.shape [104, 312, 42728]
    word_ins_out.shape = [104, 312, 43728]
    select_idx.shape = [104, 312, 30] = tgt_tokens.unsqueeze(1).expand(-1, prelen, -1)
    """
    # logits = torch.log_softmax(node_word_logits, -1, dtype=torch.float32)
    # node_word_log_prob_on_tgt_tokens = logits.gather(dim=-1, index=select_idx)
    node_word_log_prob_on_tgt_tokens = node_word_logits.gather(dim=-1, index=select_idx)
    return node_word_log_prob_on_tgt_tokens

def torch_dag_loss(match_all, links, output_length, target_length):
    r"""
    Function to calculate the dag loss.
    Input:
        match_all (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_target_length, max_output_length]
            match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
            based on the j-th vertex.
            (Note: float32 are preferred; float16 may cause precision problem)
        links (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_output_length, max_transition_length]
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
        output_length (torch.LongTensor):
            Shape: [batch_size]
            output_length should be the graph size, the vertices (index >= graph size) are ignored
        target_length (torch.LongTensor):
            Shape: [batch_size]
            target_length is the reference length, the tokens (index >= target length) are ignored

    Output (torch.FloatTensor or torch.HalfTensor):
        Shape: [batch_size]
        the loss of each sample
    """
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"

    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    match_all_chunk = torch.chunk(match_all, tarlen, -1) # k * [batch * prelen * 1]

    for k in range(1, tarlen):
        f_now = loop_function_noempty(f_arr[-1], links, match_all_chunk[k])
        f_arr.append(f_now)

    loss_result = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return loss_result

def loop_function_noempty(last_f: torch.Tensor, links: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
    f_next = logsumexp_keepdim(last_f + links, 1) # batch * 1 * prelen
    f_next = f_next.transpose(1, 2) + match # batch * prelen * 1
    return f_next


def logsumexp_keepdim(x: torch.Tensor, dim: int) -> torch.Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float('inf')
    m = m.detach()
    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    return s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, -float('inf'))

def loop_function_noempty_max(last_f: torch.Tensor, links: torch.Tensor, match: torch.Tensor) -> torch.Tensor:
    f_next = torch.max(last_f + links, dim=1)[0] # batch * 1 * prelen
    f_next = f_next.unsqueeze(-1) + match # batch * prelen * 1
    return f_next

def __torch_max_loss(match_all, links, output_length, target_length):
    match_all = match_all.transpose(1, 2)
    batch_size, prelen, tarlen = match_all.shape
    assert links.shape[1] == links.shape[2], "links should be batch_size * prelen * prelen"

    f_arr = []
    f_init = torch.zeros(batch_size, prelen, 1, dtype=match_all.dtype, device=match_all.device).fill_(float("-inf"))
    f_init[:, 0, 0] = match_all[:, 0, 0]
    f_arr.append(f_init)

    r"""
    f_init.shape = [104, 312, 1]
    match_all.shape = [104, 312, 30]
    len(match_arr) = 30
    The code below **accumulates** the probabilities of target tokens
    """

    match_arr = torch.chunk(match_all, tarlen, -1)
    for i in range(1, tarlen):
        f_now = loop_function_noempty_max(f_arr[-1], links, match_arr[i])
        f_arr.append(f_now)

    alllogprob = torch.cat(f_arr, -1)[range(batch_size), output_length - 1, target_length - 1]

    return alllogprob

def torch_dag_best_alignment(match_all, links, output_length, target_length):
    r"""
    Function to obtain the alignment between prediction and reference
    Input:
        match_all (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_target_length, max_output_length]
            match_all[b, i, j] represents -log P(y_i| v_j), the probability of predicting the i-th token in the reference
            based on the j-th vertex.
            (Note: float32 are preferred; float16 may cause precision problem)
        links (torch.FloatTensor or torch.HalfTensor):
            Shape: [batch_size, max_output_length, max_transition_length]
            links[b, i, j] represents the transition probability from the i-th vertex to **the j-th vertex**.
            (Note: this parameter is different from the cuda version)
        output_length (torch.LongTensor):
            Shape: [batch_size]
            output_length should be the graph size, the vertices (index >= graph size) are ignored
        target_length (torch.LongTensor):
            Shape: [batch_size]
            target_length is the reference length, the tokens (index >= target length) are ignored

    Output (torch.LongTensor):
        Shape: [batch_size, max_output_length]
        if output[b, i]>=0, it represents the index of target token aligned with the i-th vertex
        otherwise, output[b, i] = -1, it represents the i-th vertex is not aligned with any target token
    """
    with torch.enable_grad():
        match_all.requires_grad_()
        alllogprob = __torch_max_loss(match_all, links, output_length, target_length)
        matchgrad = torch.autograd.grad(alllogprob.sum(), [match_all]) # batch * talen * prelen
        matchgrad = matchgrad[0]
    pathvalue, path = matchgrad.max(dim=1)
    path.masked_fill_(pathvalue < 0.5, -1)
    return path

if __name__ == '__main__':
    print("Using device:", device)
    links = torch.rand((bs, L, L), device=device).triu_(1).log_()
    # links = (links/torch.sum(links, dim=-1)[:,:,None]).log() # links[b][u][v] = log transition probability from node u to node v, in batche b
    node_token_emit_prob = torch.log(torch.rand((bs, L, V), device=device)) # node_word_emit_prob[b][i][j] = log emission probability of token j from node i, in batch b
    labels = torch.randint(low=0, high=V, size=(bs, M), device=device) # indices of ground-truth tokens
    # tgt_lengths = torch.randint(low=3, high=M, size=[bs], device=device)
    tgt_lengths = torch.zeros(bs, dtype=torch.long).fill_(M)
    # tgt_lengths = torch.randint(low=3, high=M, size=[bs], device=device)
    # for b_idx in range(bs):
        # labels[b_idx, tgt_lengths[b_idx]:] = PAD_TOKEN

    # # The aim is to compute P(Y,A|X), which requires marginalization over all paths A
    # # brute-force: directly marginalize over A. P(Y, A|X) = sum_a P(Y, A=a|X); Complexity = O(L^M) 
    # for b_idx in range(bs):
    #     print(f"For batch {b_idx}, target length={tgt_lengths[b_idx]}:")
    #     all_paths = get_all_seq_of_length(L, tgt_lengths[b_idx])
    # # print(all_paths)
    #     print(f"Brute-force: sum over {len(all_paths)} paths")
    #     bf_res, path_llk_dict = brute_force_compute_likelihood_one_batch(links[b_idx], node_token_emit_prob[b_idx], all_paths, labels[b_idx], tgt_lengths[b_idx])
    #     print("Brute-force accumulated log-likelihood:", bf_res)
    #     best_path = sorted(path_llk_dict.items(), key=lambda x: x[1], reverse=True)[0]
    #     print("Brute-force best path", best_path)

    start = time.time()
    dp_res, f = dp_compute_likelihood(links, node_token_emit_prob, tgt_lengths, labels)
    end = time.time()
    print(f"(bs={bs}, M={M}, L={L}, V={V}); Use DP to calculate log-likelihood takes {end-start:.5f} seconds")
    print("Dynamic Programming accumulated log-likelihood:", dp_res)

    match_all = torch_dag_logsoftmax_gather_inplace(node_token_emit_prob, labels.unsqueeze(1).expand(-1, L, -1))
    match_all.transpose_(1, 2)

    # target_length = torch.sum(labels.ne(PAD_TOKEN), dim=-1)
    output_length = torch.zeros(bs, dtype=torch.long).fill_(L) 

    start = time.time()
    loss_result = torch_dag_loss(match_all, links, output_length, tgt_lengths)
    end = time.time()
    print(f"(bs={bs}, M={M}, L={L}, V={V}); Use DA-Transform DP impl to calculate log-likelihood takes {end-start:.5f} seconds")
    print("DA-Transformer DP impl accumulated log-likelihood:", loss_result)

    dp_best_path_llk, dp_best_path = viterbi_best_path_recursive(links[0], node_token_emit_prob[0], labels[0], torch.LongTensor([0]), torch.LongTensor([0]), torch.LongTensor([M]), torch.zeros(M, dtype=torch.long))
    dag_best_path = torch_dag_best_alignment(match_all, links, output_length, tgt_lengths)
    print("Dynamic Programming best path and best log-likelihood:", dp_best_path_llk, dp_best_path)
    print("Dynamic Programming best path (DATransformer implementation)", torch.nonzero(dag_best_path+1))

    # viterbi_llk, viterbi_best_path = viterbi_best_path_batch(links, node_token_emit_prob, labels)
    # print("Viterbi Alignment best log-likelihood & path:", viterbi_llk, viterbi_best_path)

    # # Get aligned path
    # path_arr = []
    # for path in viterbi_best_path:
    #     path_arr.append(F.pad(torch.tensor(path), (0, M-len(path)), "constant", PAD_TOKEN))
    # aligned_node_indices = torch.vstack(path_arr)

    pass
    


    
    
    