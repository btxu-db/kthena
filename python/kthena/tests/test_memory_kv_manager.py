# Copyright The Volcano Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import AsyncMock, MagicMock

import asyncio

import pytest

from kthena.runtime.kv_cache_manager import compute_standardized_hash
from kthena.runtime.memory_kv_manager import (
    KV_EVENT_CLEARED,
    KV_EVENT_REMOVED,
    KV_EVENT_SNAPSHOT,
    KV_EVENT_STORED,
    KV_EVENTS_PATH,
    MemoryKVCacheManager,
)
from kthena.runtime.router_registry import RouterRegistry


def _make_manager(endpoints):
    registry = RouterRegistry()
    for i, endpoint in enumerate(endpoints):
        registry.register(f"router-{i}", endpoint, ttl_seconds=60)

    client = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    client.post = AsyncMock(return_value=response)
    client.aclose = AsyncMock()
    return MemoryKVCacheManager(registry=registry, client=client), client


def _pushed_payloads(client):
    return [(call.args[0], call.kwargs["json"]) for call in client.post.await_args_list]


@pytest.mark.asyncio
async def test_add_blocks_pushes_standardized_hashes_to_all_routers():
    manager, client = _make_manager(
        ["http://router-a:9080", "http://router-b:9080"])

    token_ids = list(range(32))  # 2 blocks of 16
    engine_hashes = [111, 222]
    expected_std = [
        compute_standardized_hash(token_ids[0:16]),
        compute_standardized_hash(token_ids[16:32]),
    ]

    ok = await manager.add_blocks("qwen", engine_hashes, "pod-1.default", token_ids)
    assert ok
    await manager.flush()

    payloads = _pushed_payloads(client)
    assert len(payloads) == 2
    urls = {url for url, _ in payloads}
    assert urls == {
        f"http://router-a:9080{KV_EVENTS_PATH}",
        f"http://router-b:9080{KV_EVENTS_PATH}",
    }
    for _, payload in payloads:
        assert payload["pod_identifier"] == "pod-1.default"
        assert payload["model_name"] == "qwen"
        assert len(payload["events"]) == 1
        event = payload["events"][0]
        assert event["type"] == KV_EVENT_STORED
        assert event["block_hashes"] == expected_std
        assert event["timestamp"] > 0

    # engine -> std mapping is retained for later removals
    assert manager.hash_mapping == {111: expected_std[0], 222: expected_std[1]}
    await manager.close()


@pytest.mark.asyncio
async def test_remove_blocks_uses_engine_hash_mapping():
    manager, client = _make_manager(["http://router-a:9080"])
    token_ids = list(range(16))
    await manager.add_blocks("qwen", [111], "pod-1.default", token_ids)
    await manager.flush()
    client.post.reset_mock()

    removed = await manager.remove_blocks("qwen", [111, 999], "pod-1.default")
    assert removed == 1
    await manager.flush()

    payloads = _pushed_payloads(client)
    assert len(payloads) == 1
    event = payloads[0][1]["events"][0]
    assert event["type"] == KV_EVENT_REMOVED
    assert event["block_hashes"] == [compute_standardized_hash(token_ids)]
    assert 111 not in manager.hash_mapping
    await manager.close()


@pytest.mark.asyncio
async def test_shared_std_hash_survives_until_last_engine_hash_removed():
    """Blocks with identical token content but different prefixes have
    distinct engine hashes mapping to one standardized hash; the standardized
    hash must only be removed with its last engine-hash reference."""
    manager, client = _make_manager(["http://router-a:9080"])
    token_ids = list(range(16)) * 2  # both blocks share the same content
    std_hash = compute_standardized_hash(list(range(16)))

    await manager.add_blocks("qwen", [111, 222], "pod-1.default", token_ids)
    await manager.flush()
    client.post.reset_mock()

    # Removing one reference keeps the block indexed and pushes nothing.
    removed = await manager.remove_blocks("qwen", [111], "pod-1.default")
    assert removed == 1
    await manager.flush()
    client.post.assert_not_awaited()
    assert std_hash in manager._blocks["qwen"]

    # Removing the last reference drops the block and announces it.
    removed = await manager.remove_blocks("qwen", [222], "pod-1.default")
    assert removed == 1
    await manager.flush()

    payloads = _pushed_payloads(client)
    assert len(payloads) == 1
    event = payloads[0][1]["events"][0]
    assert event["type"] == KV_EVENT_REMOVED
    assert event["block_hashes"] == [std_hash]
    assert std_hash not in manager._blocks["qwen"]
    await manager.close()


@pytest.mark.asyncio
async def test_remove_blocks_without_mapping_pushes_nothing():
    manager, client = _make_manager(["http://router-a:9080"])
    removed = await manager.remove_blocks("qwen", [12345], "pod-1.default")
    assert removed == 0
    await manager.flush()
    client.post.assert_not_awaited()
    await manager.close()


@pytest.mark.asyncio
async def test_clear_all_blocks_pushes_cleared_event():
    manager, client = _make_manager(["http://router-a:9080"])
    await manager.add_blocks("qwen", [111], "pod-1.default", list(range(16)))
    await manager.flush()
    client.post.reset_mock()

    cleared = await manager.clear_all_blocks("qwen", "pod-1.default")
    assert cleared == 1
    await manager.flush()

    payloads = _pushed_payloads(client)
    assert payloads[0][1]["events"][0]["type"] == KV_EVENT_CLEARED
    assert manager.hash_mapping == {}
    await manager.close()


@pytest.mark.asyncio
async def test_push_snapshot_sends_full_index_to_single_router():
    manager, client = _make_manager(["http://router-a:9080"])
    token_ids = list(range(32))
    await manager.add_blocks("qwen", [111, 222], "pod-1.default", token_ids)
    await manager.flush()
    client.post.reset_mock()

    await manager.push_snapshot("http://router-new:9080", "pod-1.default")

    payloads = _pushed_payloads(client)
    assert len(payloads) == 1
    url, payload = payloads[0]
    assert url == f"http://router-new:9080{KV_EVENTS_PATH}"
    event = payload["events"][0]
    assert event["type"] == KV_EVENT_SNAPSHOT
    assert sorted(event["block_hashes"]) == sorted([
        compute_standardized_hash(token_ids[0:16]),
        compute_standardized_hash(token_ids[16:32]),
    ])
    # Snapshots must preserve the original per-block store times so the
    # router's engine-restart freshness filter keeps working.
    stored = manager._blocks["qwen"]
    assert event["timestamps"] == [stored[h] for h in event["block_hashes"]]
    await manager.close()


@pytest.mark.asyncio
async def test_push_snapshot_represents_cleared_model_as_empty():
    manager, client = _make_manager(["http://router-a:9080"])
    await manager.add_blocks("qwen", [111], "pod-1.default", list(range(16)))
    await manager.clear_all_blocks("qwen", "pod-1.default")
    await manager.flush()
    client.post.reset_mock()

    await manager.push_snapshot("http://router-new:9080", "pod-1.default")

    payloads = _pushed_payloads(client)
    assert len(payloads) == 1
    event = payloads[0][1]["events"][0]
    assert event["type"] == KV_EVENT_SNAPSHOT
    assert event["block_hashes"] == []
    await manager.close()


@pytest.mark.asyncio
async def test_push_snapshot_splits_large_index_into_batches(monkeypatch):
    """A snapshot larger than one batch is sent as a snapshot event replacing
    the router's view followed by stored events appending to it."""
    monkeypatch.setattr(
        "kthena.runtime.memory_kv_manager.SNAPSHOT_BATCH_SIZE", 2)
    manager, client = _make_manager(["http://router-a:9080"])
    await manager.add_blocks(
        "qwen", [101, 102, 103, 104, 105], "pod-1.default", list(range(80)))
    await manager.flush()
    client.post.reset_mock()

    await manager.push_snapshot("http://router-new:9080", "pod-1.default")

    payloads = _pushed_payloads(client)
    events = [payload["events"][0] for _, payload in payloads]
    assert [event["type"] for event in events] == [
        KV_EVENT_SNAPSHOT, KV_EVENT_STORED, KV_EVENT_STORED]
    assert [len(event["block_hashes"]) for event in events] == [2, 2, 1]
    stored = manager._blocks["qwen"]
    sent = [h for event in events for h in event["block_hashes"]]
    assert sorted(sent) == sorted(stored)
    for event in events:
        assert event["timestamps"] == [stored[h] for h in event["block_hashes"]]
    assert not manager.is_dirty("http://router-new:9080")
    await manager.close()


@pytest.mark.asyncio
async def test_push_snapshot_starts_each_model_with_snapshot_event(monkeypatch):
    monkeypatch.setattr(
        "kthena.runtime.memory_kv_manager.SNAPSHOT_BATCH_SIZE", 1)
    manager, client = _make_manager(["http://router-a:9080"])
    await manager.add_blocks("qwen", [101, 102], "pod-1.default", list(range(32)))
    await manager.add_blocks("llama", [201, 202], "pod-1.default", list(range(32, 64)))
    await manager.flush()
    client.post.reset_mock()

    await manager.push_snapshot("http://router-new:9080", "pod-1.default")

    types_by_model = {}
    for _, payload in _pushed_payloads(client):
        types_by_model.setdefault(payload["model_name"], []).append(
            payload["events"][0]["type"])
    assert types_by_model == {
        "qwen": [KV_EVENT_SNAPSHOT, KV_EVENT_STORED],
        "llama": [KV_EVENT_SNAPSHOT, KV_EVENT_STORED],
    }
    await manager.close()


@pytest.mark.asyncio
async def test_failed_snapshot_batch_stops_model_and_marks_dirty(monkeypatch):
    monkeypatch.setattr(
        "kthena.runtime.memory_kv_manager.SNAPSHOT_BATCH_SIZE", 1)
    manager, client = _make_manager(["http://router-a:9080"])
    await manager.add_blocks(
        "qwen", [101, 102, 103], "pod-1.default", list(range(48)))
    await manager.flush()
    client.post.reset_mock()

    ok_response = MagicMock()
    ok_response.raise_for_status = MagicMock()
    client.post.side_effect = [ok_response, RuntimeError("connection reset")]

    await manager.push_snapshot("http://router-new:9080", "pod-1.default")

    # The third batch is not sent: the next heartbeat restarts from a snapshot.
    events = [payload["events"][0] for _, payload in _pushed_payloads(client)]
    assert [event["type"] for event in events] == [
        KV_EVENT_SNAPSHOT, KV_EVENT_STORED]
    assert manager.is_dirty("http://router-new:9080")
    await manager.close()


@pytest.mark.asyncio
async def test_failed_push_marks_endpoint_dirty_until_snapshot_succeeds():
    manager, client = _make_manager(["http://router-a:9080"])
    client.post.side_effect = RuntimeError("connection refused")

    await manager.add_blocks("qwen", [111], "pod-1.default", list(range(16)))
    await manager.flush()
    assert manager.is_dirty("http://router-a:9080")

    # A successful snapshot reconciles the endpoint and clears dirty state.
    client.post.side_effect = None
    await manager.push_snapshot("http://router-a:9080", "pod-1.default")
    assert not manager.is_dirty("http://router-a:9080")
    await manager.close()


@pytest.mark.asyncio
async def test_failed_snapshot_keeps_endpoint_dirty():
    manager, client = _make_manager(["http://router-a:9080"])
    await manager.add_blocks("qwen", [111], "pod-1.default", list(range(16)))
    await manager.flush()
    client.post.side_effect = RuntimeError("connection refused")

    await manager.push_snapshot("http://router-a:9080", "pod-1.default")
    assert manager.is_dirty("http://router-a:9080")
    await manager.close()


@pytest.mark.asyncio
async def test_queue_overflow_marks_endpoint_dirty_and_drops_deltas(monkeypatch):
    """A router that cannot keep up must not stall event processing or grow
    memory without bound: its queue is bounded, overflowing deltas are dropped,
    and the endpoint is marked dirty for snapshot recovery."""
    monkeypatch.setattr(
        "kthena.runtime.memory_kv_manager.ENDPOINT_QUEUE_MAXSIZE", 1)
    manager, client = _make_manager(["http://router-a:9080"])

    # The router hangs on every push, so queued deltas are never drained.
    stall = asyncio.Event()

    async def hanging_post(*args, **kwargs):
        await stall.wait()

    client.post = hanging_post

    for i in range(3):
        ok = await manager.add_blocks(
            "qwen", [100 + i], "pod-1.default", list(range(16)))
        assert ok  # event processing itself is never blocked

    assert manager.is_dirty("http://router-a:9080")
    stall.set()
    await manager.close()


@pytest.mark.asyncio
async def test_no_registered_routers_skips_push():
    manager, client = _make_manager([])
    ok = await manager.add_blocks("qwen", [111], "pod-1.default", list(range(16)))
    assert ok
    await manager.flush()
    client.post.assert_not_awaited()
    await manager.close()


@pytest.mark.asyncio
async def test_add_blocks_with_mismatched_tokens_fails():
    manager, client = _make_manager(["http://router-a:9080"])
    # 10 tokens cannot be split evenly across 3 hashes
    ok = await manager.add_blocks("qwen", [1, 2, 3], "pod-1.default", list(range(10)))
    assert not ok
    client.post.assert_not_awaited()


def test_router_registry_register_and_expire(monkeypatch):
    registry = RouterRegistry()
    clock = {"now": 1000.0}
    monkeypatch.setattr(
        "kthena.runtime.router_registry.time.monotonic", lambda: clock["now"])

    # New registration needs a snapshot.
    assert registry.register("router-a", "http://10.0.0.5:9080", ttl_seconds=60)
    # Renewal before expiry does not.
    clock["now"] += 30
    assert not registry.register("router-a", "http://10.0.0.5:9080", ttl_seconds=60)
    assert registry.active_endpoints() == ["http://10.0.0.5:9080"]

    # Changing the endpoint triggers a snapshot again.
    assert registry.register("router-a", "http://10.0.0.6:9080", ttl_seconds=60)

    # Expired registrations are pruned and re-registration needs a snapshot.
    clock["now"] += 120
    assert registry.active_endpoints() == []
    assert registry.register("router-a", "http://10.0.0.6:9080", ttl_seconds=60)

    # A new process generation (router container restart with the same pod
    # name and endpoint) triggers a snapshot again.
    assert not registry.register(
        "router-a", "http://10.0.0.6:9080", ttl_seconds=60)
    assert registry.register(
        "router-a", "http://10.0.0.6:9080", ttl_seconds=60, generation="gen-2")
    assert not registry.register(
        "router-a", "http://10.0.0.6:9080", ttl_seconds=60, generation="gen-2")


def test_router_registry_rejects_invalid_input():
    registry = RouterRegistry()
    with pytest.raises(ValueError):
        registry.register("", "http://10.0.0.5:9080")
    with pytest.raises(ValueError):
        registry.register("router-a", "ftp://10.0.0.5:9080")
    with pytest.raises(ValueError):
        registry.register("router-a", "not-a-url")
    # Only base URLs are accepted: the endpoint is concatenated with
    # KV_EVENTS_PATH, so paths, queries, fragments, and userinfo are rejected.
    with pytest.raises(ValueError):
        registry.register("router-a", "http://10.0.0.5:9080/some/path")
    with pytest.raises(ValueError):
        registry.register("router-a", "http://10.0.0.5:9080?x=1")
    with pytest.raises(ValueError):
        registry.register("router-a", "http://10.0.0.5:9080#frag")
    with pytest.raises(ValueError):
        registry.register("router-a", "http://user:pass@10.0.0.5:9080")


def test_router_registry_normalizes_trailing_slash():
    registry = RouterRegistry()
    assert registry.register("router-a", "http://10.0.0.5:9080/")
    # A heartbeat with (or without) a trailing slash is the same endpoint and
    # must not be treated as a change that schedules another snapshot.
    assert not registry.register("router-a", "http://10.0.0.5:9080")
    assert not registry.register("router-a", "http://10.0.0.5:9080/")
    assert registry.active_endpoints() == ["http://10.0.0.5:9080"]
