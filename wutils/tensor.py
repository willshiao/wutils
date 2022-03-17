import torch
import tensorly as tl
import random
from torch.linalg import norm

def create_cp(dims, rank, sparsity=None, method='rand', weights=False, return_tensor=False, noise=None, sparse_noise=True):
    # TODO: investigate performance impact of setting backend here
    tl.set_backend('pytorch')

    if method == 'rand':
        randfunc = torch.rand
    elif method == 'randn':
        randfunc = torch.randn
    else:
        raise NotImplementedError(f'Unknown random method: {method}')

    n_dims = len(dims)
    factors = [randfunc((dim, rank)) for dim in dims]

    if sparsity is not None:
        if isinstance(sparsity, float):
            sparsity = (sparsity for _ in range(n_dims))
        elif not isinstance(sparsity, list) and not isinstance(sparsity, tuple):
            raise ValueError('Sparsity parameter should either be a float or tuple/list.')

        # Sparsify factors
        for dim in range(n_dims):
            n_el = dims[dim] * rank
            to_del = round(sparsity[dim] * n_el)
            idxs = torch.tensor(random.sample(range(n_el), to_del))
            to_del.view(-1)[idxs] = 0
            # torch.randperm(n_el, device=device)[:n_select]

    ten = None
    # Add noise
    if noise is not None:
        ten = tl.cp_to_tensor((torch.ones(rank), factors))
        if (sparsity is None or not sparse_noise):
            nten = torch.randn(ten.size())
            ten += noise * (norm(ten) / norm(nten)) * nten
        else:
            flat = ten.view(-1)
            nzs = torch.nonzero(flat, as_tuple=True)[0]
            nvec = torch.randn(nzs.size(1))
            flat[nzs] += noise * (norm(ten) / norm(nten)) * nvec

    if return_tensor:
        if ten is None:
            return tl.cp_to_tensor((torch.ones(rank), factors))
        return ten
    if weights:
        return torch.ones(rank), factors
    return factors
