"""DeepSeek-V4.1 state is part of PD transfer completion, not optional KV data."""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("sglang")

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.common.conn import CommonKVSender
from sglang_fl.disaggregation.conn import FlagcxKVManager, FlagcxKVSender


def _manager(state_type, *, src_item_lens=(16, 32), ratios=(2,)):
    manager = object.__new__(FlagcxKVManager)
    manager.kv_args = SimpleNamespace(
        state_types=[state_type],
        state_data_ptrs=[[1000, 2000]],
        state_item_lens=[list(src_item_lens)],
        state_dim_per_tensor=[[]],
        mla_compression_ratios=list(ratios),
        prefill_start_layer=0,
        prefill_end_layer=2,
    )
    manager.is_mla_backend = True
    manager.attn_tp_size = 8
    manager._transfer_data = Mock(return_value=0)
    return manager


def _target(*, dst_indices=(3, 4), dst_item_lens=(16, 32)):
    request = SimpleNamespace(
        flagcx_session_id="peer", dst_state_indices=[list(dst_indices)]
    )
    registration = SimpleNamespace(
        dst_state_data_ptrs=[[3000, 4000]],
        dst_state_item_lens=[list(dst_item_lens)],
        dst_state_dim_per_tensor=[[]],
        dst_attn_tp_size=8,
    )
    return request, registration


@pytest.mark.parametrize("state_type", [StateType.SWA_RING, StateType.C128_STATE])
def test_dsv41_state_uses_positional_indices_and_all_buffers(state_type):
    manager = _manager(state_type)
    request, registration = _target()
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert (
            manager.maybe_send_extra(
                request, [[1, 2]], executor, registration
            )
            == 0
        )
    manager._transfer_data.assert_called_once_with(
        "peer", [(1016, 3048, 32), (2032, 4096, 64)]
    )


@pytest.mark.parametrize("state_type", [StateType.SWA_RING, StateType.C128_STATE])
def test_dsv41_positional_state_rejects_index_mismatch(state_type):
    manager = _manager(state_type)
    request, registration = _target(dst_indices=(3,))
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert manager.maybe_send_extra(request, [[1, 2]], executor, registration) == -1
    manager._transfer_data.assert_not_called()


def test_dsv41_c2_rejects_different_receiver_stride_before_write():
    manager = _manager(StateType.C128_STATE)
    request, registration = _target(dst_item_lens=(16, 48))
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert manager.maybe_send_extra(request, [[1, 2]], executor, registration) == -1
    manager._transfer_data.assert_not_called()


def test_sender_accepts_sglang_0518_request_arguments():
    manager = Mock()
    manager.enable_all_cp_ranks_for_transfer = False
    manager.is_dummy_cp_rank = False
    with patch.object(CommonKVSender, "__init__", return_value=None) as base_init:
        sender = FlagcxKVSender(
            manager, "127.0.0.1:8998", 17, [0], 0,
            req_has_disagg_prefill_dp_rank=True,
        )
    base_init.assert_called_once_with(
        manager, "127.0.0.1:8998", 17, [0], 0, True
    )
    sender.kv_mgr = manager
    sender.bootstrap_room = 17
    sender.curr_idx = 0
    sender.num_kv_indices = 2
    sender.aux_index = 4
    sender._record_transfer_indices = Mock()
    indices = np.array([3, 4], dtype=np.int32)

    sender.send(indices, state_indices=[], num_kv_tokens=256)

    manager.add_transfer_request.assert_called_once()
    args, kwargs = manager.add_transfer_request.call_args
    assert args[:1] == (17,)
    np.testing.assert_array_equal(args[1], indices)
    assert args[2:] == (slice(0, 2), True)
    assert kwargs == {"aux_index": 4, "state_indices": []}
