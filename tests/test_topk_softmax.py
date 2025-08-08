import pytest
import sgl_kernel
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")

# TODO: use this as ref function
def fused_topk_torch_native(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    assert (
        hidden_states.shape[0] == gating_output.shape[0]
    ), f"Number of tokens mismatch, {hidden_states.shape=} vs {gating_output.shape=}"
    M, _ = hidden_states.shape
    topk_weights = torch.empty(
        M, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    topk_weights = F.softmax(gating_output.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(topk_weights, topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


def ref_topk_softmax(gating_logits, n_topk):
    gating_logits = gating_logits.to(torch.float)
    softmax = torch.nn.functional.softmax(gating_logits, dim=-1, dtype=torch.float)
    topk_weights, topk_indices = torch.topk(softmax, n_topk, dim=-1)

    # token_for_expert: # of tokens for each expert
    # token_offset: the offset of each token for each export
    n_experts = gating_logits.shape[-1]
    token_for_experts = torch.zeros(
        n_experts, device=device, dtype=torch.int32
    )
    for i in range(n_experts):
        token_for_experts[i] = (topk_indices == i).sum().item()

    return topk_weights, topk_indices, token_for_experts

    
# TODO: test more cases
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n_token", [2])
# @pytest.mark.parametrize("n_token", [2, 32, 4096])
@pytest.mark.parametrize("n_expert", [8])
# @pytest.mark.parametrize("n_expert", [8, 32])
@pytest.mark.parametrize("n_topk", [2])
# @pytest.mark.parametrize("n_topk", [1, 2, 4, 8])
@pytest.mark.parametrize("renormalize", [False])
# @pytest.mark.parametrize("renormalize", [False, True])
def test_topk_softmax(dtype, n_token, n_topk, n_expert, renormalize):
    gating_logits = torch.randn(n_token, n_expert, device=device, dtype=dtype)
    
    ref_token_weights, ref_topk_indices, ref_token_for_experts = (
        ref_topk_softmax(gating_logits, n_topk)
    )
    # TODO: currently the schema followed the one in ipex xpu. Align the schema with cuda schema in sglang.
    topk_weights, topk_indices, token_for_experts, _ = (
        sgl_kernel.topk_softmax(gating_logits, n_topk, False)
    )
    
    # Compare the results
    torch.testing.assert_close(
        ref_token_weights, topk_weights, atol=1e-2, rtol=1e-2
    )
    assert torch.equal(ref_topk_indices, topk_indices)
    assert torch.equal(ref_token_for_experts, token_for_experts)


if __name__ == "__main__":
    pytest.main([__file__])    