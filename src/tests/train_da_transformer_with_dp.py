import torch
import copy
torch.manual_seed(2023) # for reproducibility


M = 5 # target token sequence length
L = 10 # number of nodes in the DAG = graph size
bs = 2 # batch size
V = 100 # size of vocabulary 
PAD_TOKEN = -1 

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

def dp_compute_likelihood(links, node_token_emit_prob, labels):
    f = torch.zeros((bs, M, L)).fill_(float(-"inf")) # log likelihood
    # f[f==0] = float("-inf") # log likelihood
    b_idx = torch.arange(bs)
    f[b_idx, 0, 0] = node_token_emit_prob[b_idx, 0,labels[b_idx, 0]]
    for i in range(1, M):
        for u in range(i, L-(M-i)+1):
            trans_log_prob = torch.log(torch.exp(f[:,i-1,:u])[:,None,:].matmul(torch.exp(links[b_idx,:u, u])[:,:,None])).squeeze(-1).squeeze(-1)
            emit_log_prob = node_token_emit_prob[b_idx,u,labels[b_idx,i]].squeeze(-1)
            f[b_idx, i, u] = emit_log_prob +trans_log_prob
    return f[b_idx, M-1, L-1]

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
    return f, [bp[L-1][M-1] for bp in best_paths]

if __name__ == '__main__':
    links = torch.rand((bs, L, L)) 
    links = (links/torch.sum(links, dim=-1)[:,:,None]).log() # links[b][u][v] = log transition probability from node u to node v, in batche b
    node_token_emit_prob = torch.log(torch.rand((bs, L, V))) # node_word_emit_prob[b][i][j] = log emission probability of token j from node i, in batch b
    labels = torch.randint(low=0, high=V, size=(bs, M)) # indices of ground-truth tokens
    tgt_lengths = torch.randint(low=1, high=M, size=[bs])
    for b_idx in range(bs):
        labels[b_idx, tgt_lengths[b_idx]:] = PAD_TOKEN

    # The aim is to compute P(Y,A|X), which requires marginalization over all paths A
    # brute-force: directly marginalize over A. P(Y, A|X) = sum_a P(Y, A=a|X); Complexity = O(L^M) 
    all_paths = get_all_seq_of_length(L, M)
    # print(all_paths)
    print(f"Brute-force: sum over {len(all_paths)} paths")
    bf_res, path_llk_dict = brute_force_compute_likelihood(links, node_token_emit_prob, all_paths, labels)
    print("Brute-force accumulated log-likelihood:", bf_res)
    for b_idx in range(bs):
        print(f"For batch {b_idx}:")
        best_path = sorted(path_llk_dict[b_idx].items(), key=lambda x: x[1], reverse=True)[0]
        print("Brute-force best path", best_path)

    dp_res = dp_compute_likelihood(links, node_token_emit_prob, labels)
    print("Dynamic Programming accumulated log-likelihood:", dp_res)

    # dp_best_path_llk, dp_best_path = viterbi_best_path_recursive(links[0], node_token_emit_prob[0], labels[0], torch.LongTensor([0]), torch.LongTensor([0]), torch.LongTensor([M]), torch.zeros(M, dtype=torch.long))
    # print("Dynamic Programming best path and best log-likelihood:", dp_best_path, dp_best_path_llk)

    viterbi_llk, viterbi_best_path = viterbi_best_path_batch(links, node_token_emit_prob, labels)
    print("Viterbi Alignment best log-likelihood & path:", viterbi_llk, viterbi_best_path)
    pass
    


    
    
    