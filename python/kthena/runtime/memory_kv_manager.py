# Copyright The Volcano Authors.
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

import asyncio
import logging
import time
from typing import Dict, List, Optional

import httpx

from kthena.runtime.kv_cache_manager import standardize_block_hashes
from kthena.runtime.router_registry import RouterRegistry, get_router_registry

logger = logging.getLogger(__name__)

KV_EVENTS_PATH = "/kvcache/events"

KV_EVENT_STORED = "stored"
KV_EVENT_REMOVED = "removed"
KV_EVENT_CLEARED = "cleared"
KV_EVENT_SNAPSHOT = "snapshot"

PUSH_TIMEOUT_SECONDS = 2.0

# Upper bound of deliveries queued per router endpoint. When a router cannot
# keep up and its queue overflows, further deltas are dropped and the endpoint
# is marked dirty so its next registration heartbeat receives a full snapshot.
ENDPOINT_QUEUE_MAXSIZE = 1024

# Upper bound of blocks carried by one snapshot request. The router caps KV
# event request bodies at 4 MiB (kvEventsMaxBodyBytes in
# pkg/kthena-router/scheduler/plugins/kvcache_memory_index.go), and each block
# encodes to at most ~33 bytes of JSON (a 63-bit hash and a 10-digit timestamp
# plus separators), so a full snapshot of a large KV cache has to be split.
# 50000 blocks stay under ~1.6 MiB, well below the cap.
SNAPSHOT_BATCH_SIZE = 50000

# Queue item kinds processed by the per-endpoint delivery worker.
_ITEM_DELTA = "delta"
_ITEM_SNAPSHOT = "snapshot"


class MemoryKVCacheManager:
    """KV cache manager that pushes standardized block hashes directly to
    registered router instances instead of writing them to Redis.

    Implements the same interface as VLLMKVCacheRedisManager
    (add_blocks / remove_blocks / clear_all_blocks) so it can back
    VLLMKVCacheEventHandler and SGLangKVCacheEventHandler.

    Unlike Redis mode, there is no shared, durable store here: every router
    process keeps its own private in-memory copy of the index. In Redis mode
    the sidecar writes each event once into Redis and can forget it, because
    Redis itself is the authoritative state that any router — including one
    that just (re)started or briefly lost connectivity — reads on demand. In
    memory mode this class has to retain that authoritative state (_blocks)
    itself, so it can replay it as a full snapshot to any router that starts
    fresh, restarts, or missed a delta push.

    Delivery is decoupled per endpoint: each registered router has its own
    bounded queue drained by a dedicated worker task, so one slow or
    unreachable router delays neither the other routers nor the engine event
    consumer. Snapshots travel through the same queue and build their payload
    at delivery time, which keeps them ordered with the deltas around them.
    On queue overflow the endpoint is marked dirty and deltas are dropped;
    the next registration heartbeat then pushes a full snapshot, which
    supersedes everything the endpoint may have missed.
    """

    def __init__(self, registry: Optional[RouterRegistry] = None,
                 client: Optional[httpx.AsyncClient] = None):
        self.registry = registry or get_router_registry()
        self._client = client
        # engine hash -> standardized hash, needed because removal events only
        # carry engine hashes.
        self.hash_mapping: Dict[int, int] = {}
        # model name -> {std_hash: number of engine hashes currently mapped to
        # it}. Blocks with identical token content but different prefixes have
        # distinct engine hashes yet the same standardized hash, so a
        # standardized hash may only be dropped once its last engine-hash
        # reference is removed.
        self._std_refs: Dict[str, Dict[int, int]] = {}
        # model name -> {std_hash: unix seconds when stored}, mirrors what has
        # been pushed so a newly registered router can receive a snapshot.
        # Models stay present (with an empty dict) after being cleared so a
        # snapshot can authoritatively represent an empty cache.
        self._blocks: Dict[str, Dict[int, int]] = {}
        # Per-endpoint bounded delivery queues and their worker tasks.
        self._queues: Dict[str, asyncio.Queue] = {}
        self._workers: Dict[str, asyncio.Task] = {}
        # Endpoints whose last push failed or whose queue overflowed; they
        # receive a fresh snapshot on their next registration heartbeat
        # instead of staying divergent.
        self._dirty_endpoints: set = set()

    def is_dirty(self, endpoint: str) -> bool:
        """Whether this endpoint's index may have diverged (a push failed or
        deltas were dropped), requiring a fresh snapshot."""
        return endpoint in self._dirty_endpoints

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(PUSH_TIMEOUT_SECONDS))
        return self._client

    async def close(self) -> None:
        for worker in self._workers.values():
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers.values(),
                                 return_exceptions=True)
        self._workers.clear()
        self._queues.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def flush(self) -> None:
        """Wait until every queued delivery has been processed (test helper
        and shutdown aid)."""
        if self._queues:
            await asyncio.gather(*(q.join() for q in self._queues.values()))

    async def add_blocks(self, model_name: str, block_hashes: List[int],
                         pod_identifier: str, token_ids: Optional[List[int]] = None) -> bool:
        if not block_hashes or not model_name or not pod_identifier:
            return not block_hashes
        if not token_ids:
            return True

        pairs = standardize_block_hashes(block_hashes, token_ids)
        if pairs is None:
            return False

        timestamp = int(time.time())
        model_blocks = self._blocks.setdefault(model_name, {})
        model_refs = self._std_refs.setdefault(model_name, {})
        std_hashes = []
        for engine_hash, std_hash in pairs:
            previous = self.hash_mapping.get(engine_hash)
            if previous != std_hash:
                if previous is not None:
                    self._release_std_hash(model_name, previous)
                model_refs[std_hash] = model_refs.get(std_hash, 0) + 1
            self.hash_mapping[engine_hash] = std_hash
            model_blocks[std_hash] = timestamp
            if std_hash not in std_hashes:
                std_hashes.append(std_hash)

        self._push_to_all(pod_identifier, model_name, [{
            "type": KV_EVENT_STORED,
            "block_hashes": std_hashes,
            "timestamp": timestamp,
        }])
        logger.info(
            f"Runtime memory push - Model: {model_name}, Pod: {pod_identifier}, "
            f"Count: {len(std_hashes)}")
        return True

    def _release_std_hash(self, model_name: str, std_hash: int) -> bool:
        """Drop one engine-hash reference; returns True when it was the last
        one and the standardized hash left the index."""
        model_refs = self._std_refs.get(model_name, {})
        remaining = model_refs.get(std_hash, 0) - 1
        if remaining > 0:
            model_refs[std_hash] = remaining
            return False
        model_refs.pop(std_hash, None)
        self._blocks.get(model_name, {}).pop(std_hash, None)
        return True

    async def remove_blocks(self, model_name: str, block_hashes: List[int],
                            pod_identifier: str) -> int:
        if not block_hashes or not model_name or not pod_identifier:
            return 0

        removed = 0
        std_hashes = []
        for engine_hash in block_hashes:
            std_hash = self.hash_mapping.pop(engine_hash, None)
            if std_hash is None:
                continue
            removed += 1
            # Only announce the removal once the last engine hash referencing
            # this standardized hash is gone; other cached blocks with the
            # same token content may still exist.
            if self._release_std_hash(model_name, std_hash):
                std_hashes.append(std_hash)

        if not removed:
            return 0

        if std_hashes:
            self._push_to_all(pod_identifier, model_name, [{
                "type": KV_EVENT_REMOVED,
                "block_hashes": std_hashes,
            }])
        logger.info(
            f"Removed {len(std_hashes)} blocks for model {model_name}, pod {pod_identifier}")
        return removed

    async def clear_all_blocks(self, model_name: str, pod_identifier: str) -> int:
        if not model_name or not pod_identifier:
            return 0

        cleared = len(self._blocks.get(model_name, {}))
        # Keep the model key so later snapshots can still represent the empty
        # cache for this model and remove stale router entries.
        self._blocks[model_name] = {}
        self._std_refs.clear()
        self.hash_mapping.clear()

        self._push_to_all(pod_identifier, model_name, [{
            "type": KV_EVENT_CLEARED,
        }])
        logger.info(
            f"Cleared {cleared} blocks for model {model_name}, pod {pod_identifier}")
        return cleared

    async def push_snapshot(self, endpoint: str, pod_identifier: str) -> None:
        """Queue a full-index snapshot for a single router endpoint.

        Called when a router registers for the first time, re-registers after
        its previous registration expired, or heartbeats while marked dirty
        after a failed or dropped push, so it can rebuild its in-memory index.

        The snapshot is delivered by the endpoint's worker in order with the
        deltas around it, and its payload is built at delivery time, so it
        reflects every mutation whose delta precedes it and cannot be
        overtaken by a newer delta it does not contain. Pending deltas are
        discarded first: the snapshot supersedes them.

        Each model is sent in batches of at most SNAPSHOT_BATCH_SIZE blocks to
        stay under the router's request body limit: the first batch is a
        snapshot event replacing the router's view of that model, the rest
        are stored events appending to it.
        """
        queue = self._ensure_worker(endpoint)
        self._drain_queue(queue)
        queue.put_nowait((_ITEM_SNAPSHOT, pod_identifier, None, None))
        await queue.join()

    def _push_to_all(self, pod_identifier: str, model_name: str,
                     events: List[dict]) -> None:
        """Queue a delta for every registered router endpoint; delivery is
        asynchronous so a slow router never blocks event processing."""
        endpoints = self.registry.active_endpoints()
        self._prune_workers(endpoints)
        if not endpoints:
            logger.debug("No routers registered, skipping KV event push")
            return
        for endpoint in endpoints:
            queue = self._ensure_worker(endpoint)
            try:
                queue.put_nowait(
                    (_ITEM_DELTA, pod_identifier, model_name, events))
            except asyncio.QueueFull:
                # The router is too slow to keep up; drop the delta and let
                # its next registration heartbeat recover it with a snapshot.
                if endpoint not in self._dirty_endpoints:
                    logger.warning(
                        f"KV event queue for router {endpoint} is full; "
                        f"dropping deltas until a snapshot reconciles it")
                self._dirty_endpoints.add(endpoint)

    def _ensure_worker(self, endpoint: str) -> asyncio.Queue:
        queue = self._queues.get(endpoint)
        if queue is None:
            queue = asyncio.Queue(maxsize=ENDPOINT_QUEUE_MAXSIZE)
            self._queues[endpoint] = queue
            self._workers[endpoint] = asyncio.get_running_loop().create_task(
                self._deliver(endpoint, queue))
        return queue

    def _prune_workers(self, active_endpoints: List[str]) -> None:
        """Stop workers of endpoints whose registration expired."""
        active = set(active_endpoints)
        for endpoint in list(self._workers):
            if endpoint not in active:
                self._workers.pop(endpoint).cancel()
                queue = self._queues.pop(endpoint)
                self._drain_queue(queue)
                self._dirty_endpoints.discard(endpoint)

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                return

    async def _deliver(self, endpoint: str,
                       queue: asyncio.Queue) -> None:
        """Per-endpoint worker delivering queued items sequentially."""
        while True:
            kind, pod_identifier, model_name, events = await queue.get()
            try:
                if kind == _ITEM_SNAPSHOT:
                    await self._send_snapshot(endpoint, pod_identifier)
                elif endpoint in self._dirty_endpoints:
                    # This endpoint already diverged; skip the delta instead
                    # of burning a timeout on it — the pending snapshot
                    # supersedes it anyway.
                    pass
                else:
                    await self._send(
                        endpoint, pod_identifier, model_name, events)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - keep the worker alive
                self._dirty_endpoints.add(endpoint)
                logger.warning(
                    f"Unexpected error delivering KV events to router "
                    f"{endpoint}: {e}")
            finally:
                queue.task_done()

    async def _send_snapshot(self, endpoint: str, pod_identifier: str) -> None:
        ok = True
        # Copy before awaiting: events handled while a batch is in flight may
        # add models or blocks, and their deltas are queued behind this item.
        for model_name, model_blocks in list(self._blocks.items()):
            # Preserve original per-block store times; re-stamping them at
            # snapshot time would defeat the engine-restart freshness filter.
            items = list(model_blocks.items())
            # The first batch replaces what the router knows about this model
            # and the rest append to it. An empty model still sends one empty
            # snapshot so the router drops its stale entries.
            for start in range(0, max(len(items), 1), SNAPSHOT_BATCH_SIZE):
                batch = items[start:start + SNAPSHOT_BATCH_SIZE]
                if not await self._send(endpoint, pod_identifier, model_name, [{
                    "type": KV_EVENT_SNAPSHOT if start == 0 else KV_EVENT_STORED,
                    "block_hashes": [h for h, _ in batch],
                    "timestamps": [ts for _, ts in batch],
                }]):
                    # The endpoint is dirty now, so the next heartbeat restarts
                    # this model from a snapshot; the remaining batches would
                    # only be wasted.
                    ok = False
                    break
        if ok:
            self._dirty_endpoints.discard(endpoint)
            logger.info(f"Pushed KV snapshot to router endpoint {endpoint}")
        else:
            logger.warning(
                f"KV snapshot to router endpoint {endpoint} failed; will retry "
                f"on its next registration heartbeat")

    async def _send(self, endpoint: str, pod_identifier: str,
                    model_name: str, events: List[dict]) -> bool:
        payload = {
            "pod_identifier": pod_identifier,
            "model_name": model_name,
            "events": events,
        }
        try:
            response = await self._get_client().post(
                f"{endpoint}{KV_EVENTS_PATH}", json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            # Mark the endpoint dirty so its next registration heartbeat
            # triggers a full snapshot instead of leaving it divergent.
            self._dirty_endpoints.add(endpoint)
            logger.warning(f"Failed to push KV events to router {endpoint}: {e}")
            return False


_memory_kv_manager: Optional[MemoryKVCacheManager] = None


def get_memory_kv_manager() -> MemoryKVCacheManager:
    global _memory_kv_manager
    if _memory_kv_manager is None:
        _memory_kv_manager = MemoryKVCacheManager()
    return _memory_kv_manager
