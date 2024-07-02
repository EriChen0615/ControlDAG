import torch
import torch.nn.functional as F

def torch_tensor_intersect(a, b):
    ac = a.detach().clone()
    bc = b.detach().clone()
    batch_size = a.shape[0]
    intersects = []
    for b_idx in range(batch_size):
        aa, bb = torch.unique(ac[b_idx]), torch.unique(bc[b_idx])
        aa, bb = pad_last_dim_to_same_shape(aa, bb)
        a_cat_b, counts = torch.cat([aa,bb]).unique(return_counts=True)
        intersected_tokens = a_cat_b[torch.where(counts.gt(1))] 
        intersects.append(intersected_tokens.tolist() if len(intersected_tokens) else [0])
    return intersects

def pad_last_dim_to_same_shape(a, b):
    assert len(a.shape) == len(b.shape), f"to pad to same shape, no broadcasting is allowed: a.shape={a.shape}, b.shape={b.shape}"
    max_len = max(a.shape[-1], b.shape[-1])
    pad_shape = [0, 0] * len(a.shape)
    if a.shape[-1] < b.shape[-1]:
        pad_shape[1] = max_len - a.shape[-1]
        a = F.pad(a, pad_shape)
    else:
        pad_shape[1] = max_len - b.shape[-1]
        b = F.pad(b, pad_shape)
    return a, b

def unique_rows_indices(tensor):
    # Create a view of the tensor where each row is treated as a single item
    tensor_view = tensor.contiguous().view(tensor.size(0), -1)
    # Create a unique tensor
    _, unique_indices = torch.unique(tensor_view, dim=0, return_inverse=True)
    return unique_indices