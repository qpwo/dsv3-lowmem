import torch

def top_k_uniq(vecs: torch.Tensor, k: int, max_uniq: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get top-k values with a maximum number of unique indices (across the entire batch).
    The output shape is (num_vecs, k), just like torch.topk(vecs, k, dim=-1).
    """
    num_vecs, vec_dim = vecs.shape

    # Flatten batch dimension to get global top values
    flat_vecs = vecs.reshape(-1)
    _, global_indices = torch.topk(flat_vecs, min(max_uniq, len(flat_vecs)))

    # Create mask for allowed indices
    allowed_indices = global_indices % vec_dim
    mask = torch.zeros(vec_dim, dtype=torch.bool, device=vecs.device)
    mask[allowed_indices] = True

    # Apply mask to original vectors
    masked_vecs = torch.where(mask, vecs, torch.full_like(vecs, float('-inf')))

    # Get top-k values from masked vectors
    values, indices = torch.topk(masked_vecs, k, dim=-1)

    return values, indices


def _test_top_k_uniq():
    num_vecs = 10
    vec_dim = 100
    k = 5
    max_uniq = 7
    vecs = torch.rand(num_vecs, vec_dim) * torch.rand(1, vec_dim)
    idxs1 = torch.topk(vecs, k, dim=-1)[1]
    idxs2 = top_k_uniq(vecs, k, max_uniq=max_uniq)[1]
    assert idxs1.shape == idxs2.shape == (num_vecs, k)
    assert len(set(idxs2.flatten().tolist())) <= max_uniq
    print(f"{k=} {max_uniq=}")
    print(f"{idxs1=}")
    print(f"{idxs2=}")


if __name__ == '__main__':
    _test_top_k_uniq()
