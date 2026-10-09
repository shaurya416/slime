import base64
from argparse import Namespace

import numpy as np
import pytest
import torch

from slime.observability import rollout_metrics
from slime.observability.rollout_metrics import _compute_top_p_kept_vocab_metrics, log_eval_rollout_data
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.misc import decode_int32_meta_array
from slime.utils.types import Sample

NUM_GPUS = 0


def _make_args():
    return Namespace(sglang_speculative_algorithm=False, num_layers=2, moe_router_topk=2)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(samples) == {}


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


@pytest.mark.unit
def test_decode_int32_meta_array_decodes_base64_to_tensor():
    decoded = decode_int32_meta_array({"routed_experts": _b64_int32([1, 2, 3])}, "routed_experts")

    assert torch.is_tensor(decoded)
    assert decoded.dtype == torch.int32
    torch.testing.assert_close(decoded, torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.tokens == [0, 1, 10, 20]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
    torch.testing.assert_close(
        sample.rollout_routed_experts,
        torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_append_response_tokens_ignores_split_pd_routed_experts():
    sample = Sample(tokens=[101, 102, 103, 104])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "pd_prefill_routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "pd_decode_routed_experts": _b64_int32([8, 9, 10, 11]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts is None


@pytest.mark.unit
def test_append_response_tokens_rejects_mismatched_routed_experts_shape():
    sample = Sample(tokens=[101, 102, 103])

    with pytest.raises(ValueError, match="routed_experts element count"):
        sample.append_response_tokens(
            _make_args(),
            tokens=[],
            trainable=True,
            meta_info={
                "routed_experts": _b64_int32([0, 1, 2, 3]),
                "finish_reason": {"type": "stop"},
            },
        )


@pytest.mark.unit
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.tokens == [0, 1, 200, 201, 202]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_requires_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="trainable response tokens require rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], trainable=True)


@pytest.mark.unit
def test_append_response_tokens_rejects_non_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="non-trainable response tokens should not pass rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], log_probs=[-0.1], trainable=False)


@pytest.mark.unit
def test_eval_passrate_uses_each_dataset_group_size(monkeypatch):
    monkeypatch.setattr(rollout_metrics.logging_utils, "log", lambda *args, **kwargs: None)
    args = Namespace(
        custom_eval_rollout_log_function_path=None,
        log_passrate=True,
        n_samples_per_eval_prompt=8,
        wandb_always_use_train_step=False,
        eval_datasets=[
            EvalDatasetConfig(name="sampled", path="unused.jsonl", n_samples_per_eval_prompt=4),
            EvalDatasetConfig(name="greedy", path="unused.jsonl", n_samples_per_eval_prompt=1),
        ],
    )
    data = {
        # 4 prompts x 4 samples; only the first prompt is solved.
        "sampled": {"rewards": [1, 1, 1, 1] + [0] * 12},
        # 10 prompts x 1 sample.
        "greedy": {"rewards": [1, 0] * 5},
    }

    log_dict = log_eval_rollout_data(0, args, data)

    assert log_dict["eval/sampled-pass@1"] == pytest.approx(0.25)
    assert log_dict["eval/sampled-pass@4"] == pytest.approx(0.25)
    assert "eval/sampled-pass@8" not in log_dict
    assert log_dict["eval/greedy"] == pytest.approx(0.5)
    assert not any(key.startswith("eval/greedy-pass@") for key in log_dict)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
