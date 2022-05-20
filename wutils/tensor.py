import torch
import tensorly as tl
import random
from torch.linalg import norm

# Implementation of Tensor Toolbox's normalize() function
# NOT the same as Tensorly's cp_normalize function
def normalize_cp_ten(cp_ten, norm_type=1, inplace=True):
    lambdas, U = cp_ten
    ndims = len(U)

    if not inplace:
        lambdas = lambdas.clone()
        U = [x.clone() for x in U]


    for r in range(lambdas.size(0)):
        for n in range(ndims):
            tmp = torch.norm(U[n][:, r], p=norm_type)
            if tmp > 0:
                U[n][:, r] /= tmp
            lambdas[r] *= tmp

    # Fix negative values
    idx = torch.where(lambdas <= 0)
    if idx[0].size(0) > 0:
        U[0][:, idx] *= -1
        lambdas[idx] *= -1

    return (lambdas, U)


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
            sparsity = [sparsity for _ in range(n_dims)]
        elif not isinstance(sparsity, list) and not isinstance(sparsity, tuple):
            raise ValueError('Sparsity parameter should either be a float or tuple/list.')

        # Sparsify factors
        for dim in range(n_dims):
            n_el = dims[dim] * rank
            to_del = round(sparsity[dim] * n_el)
            if to_del == 0:
                continue
            idxs = torch.tensor(random.sample(range(n_el), to_del))
            factors[dim].view(-1)[idxs] = 0
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
            nvec = torch.randn(nzs.size(0))
            flat[nzs] += noise * (norm(ten) / norm(nvec)) * nvec

    if return_tensor:
        if ten is None:
            return tl.cp_to_tensor((torch.ones(rank), factors))
        return ten
    if weights:
        return torch.ones(rank), factors
    return factors

# Proportional sampling, like in the Tensor Toolbox
# If return_counts=True: equivalent to accumarray(prosample(n_samples, prob), 1, [rank 1]) in MATLAB
def probsample(n_samples, prob, return_counts=True):
    if return_counts:
        bins = torch.cat([torch.zeros(1), torch.cumsum(prob, dim=0)])
        bins[-1] = 1
        res = torch.histogram(torch.rand(n_samples), bins)
        return res.hist

    bins = torch.cumsum(prob, dim=0)
    bins[-1] = 1
    return torch.bucketize(torch.rand((n_samples,)), bins)

def create_cp_sparse_gen(dims, rank, n_el, method='rand', return_sparse=False):
    tl.set_backend('pytorch')

    if method == 'rand':
        randfunc = torch.rand
    elif method == 'randn':
        randfunc = torch.randn
    else:
        raise NotImplementedError(f'Unknown random method: {method}')

    n_dims = len(dims)
    factors = [randfunc((dim, rank)) for dim in dims]
    lambdas = torch.ones(rank)

    # Create probability tensor
    P = normalize_cp_ten((lambdas, factors))
    lambdas /= torch.sum(lambdas)

    # Count samples per component
    n_edges = n_el
    if n_edges < 1:
        n_edges = round(n_edges * torch.prod(dims))
    csums = probsample(n_edges, lambdas)

    subs = []
    # Calculate subscripts
    for c in range(rank):
        n_sample = int(csums[c])
        if n_sample == 0:
            continue

        sub_idxs = torch.zeros((int(n_sample), n_dims), dtype=torch.int64)
        for d in range(n_dims):
            sub_idxs[:, d] = probsample(n_sample, factors[d][:, c], return_counts=False)
        subs.append(sub_idxs)

    all_subs = torch.vstack(subs).T

    sp_ten = torch.sparse_coo_tensor(all_subs, torch.ones(n_edges))
    if return_sparse:
        return sp_ten

    return sp_ten.to_dense()
