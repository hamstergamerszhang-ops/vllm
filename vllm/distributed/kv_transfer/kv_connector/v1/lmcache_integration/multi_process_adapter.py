# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice
from typing import Any

import torch
import zmq
from lmcache.utils import _lmcache_nvtx_annotate, init_logger
from lmcache.v1.multiprocess.custom_types import (
    CudaIPCWrapper,
    IPCCacheServerKey,
    KVCache,
)
from lmcache.v1.multiprocess.mq import MessageQueueClient, MessagingFuture
from lmcache.v1.multiprocess.protocol import RequestType, get_response_class

logger = init_logger(__name__)


def wrap_kv_caches(kv_caches: dict[str, torch.Tensor]) -> KVCache:
    logger.info("KV caches keys are %s", list(kv_caches.keys()))
    return [CudaIPCWrapper(tensor) for tensor in kv_caches.values()]


def striding_block_hashes(
    block_hashes: list[bytes], blocks_in_chunk: int
) -> Iterable[bytes]:
    """Extract chunk-level hashes from block hashes by striding.

    In hash-based vLLM, each vLLM block has its own hash.  LMCache chunks
    span ``blocks_in_chunk`` consecutive blocks.  The representative hash
    for a chunk is the hash of the **last** block in that chunk (because
    each block hash already encodes its prefix).  So we start at index
    ``blocks_in_chunk - 1`` and stride by ``blocks_in_chunk``.
    """
    return islice(block_hashes, blocks_in_chunk - 1, None, blocks_in_chunk)


def send_lmcache_request(
    mq_client: MessageQueueClient,
    request_type: RequestType,
    payloads: list[Any],
) -> MessagingFuture[Any]:
    """
    Helper function to send the request to the LMCache multiprocess server

    Args:
        mq_client: The LMCache multiprocess mode message queue client
        request_type: The request type
        payloads: The request payloads

    Returns:
        A messaging future for the request
    """

    future = mq_client.submit_request(
        request_type, payloads, get_response_class(request_type)
    )
    return future


def get_lmcache_chunk_size(
    mq_client: MessageQueueClient,
) -> int:
    """
    Helper function to get the LMCache chunk size from the server

    Args:
        mq_client: The LMCache multiprocess mode message queue client

    Returns:
        An integer representing the LMCache chunk size
    """
    future = send_lmcache_request(mq_client, RequestType.GET_CHUNK_SIZE, [])
    chunk_size = future.result()
    return chunk_size


def _device_vendor() -> str:
    """Return the GPU vendor string for cache keys.

    Mirrors the LMCache-tree adapter's
    ``"rocm" if _is_rocm() else "nvidia" if _has_cuda() else ""``.
    The server uses this for vendor-specific routing; an empty string is
    the safe non-GPU fallback.
    """
    try:
        if hasattr(torch, "version") and torch.version.hip:  # ROCm
            return "rocm"
        if torch.cuda.is_available():
            return "nvidia"
    except Exception:
        pass
    return ""


# NOTE: This dataclass is synced from
# ``lmcache.integration.vllm.vllm_multi_process_adapter.ParallelStrategy``
# so the vendored fallback computes the same KV-parallel geometry (MLA
# reader/writer selection, per-server reader counts, DP/DCP/PCP offsets)
# as the LMCache-tree adapter.  The old stored ``kv_world_size`` /
# ``kv_worker_id`` / ``actual_world_size`` / ``actual_worker_id`` fields
# are replaced by computed properties derived from the raw vLLM parallel
# config; this is what makes pipeline parallelism correct for MLA models
# (see LMCache PR #4082).  When the ``lmcache`` package is importable the
# LMCache-tree class is used instead of this vendored copy.
@dataclass
class ParallelStrategy:
    use_mla: bool
    """Whether to use the MLA."""

    vllm_world_size: int
    """Number of workers managed by one vLLM scheduler (TP × PP; excludes DP).

    Mirrors ``vllm.parallel_config.world_size``.
    """

    vllm_worker_id: int
    """This worker's rank within its scheduler group."""

    tp_size: int
    """The tensor parallel size."""

    pp_size: int
    """The pipeline parallel size."""

    n_servers: int
    """Number of LMCache servers backing this deployment"""

    dp_rank: int = 0
    """Data-parallel rank (0 when dp_size=1).  Used to offset server
    block assignment so each DP replica targets different servers."""

    dp_size: int = 1
    """Data-parallel size.  When >1, each DP replica gets
    ``n_servers // dp_size`` servers."""

    dcp_size: int = 1
    """Decode context parallel size.  DCP splits a TP group into
    ``tp_size // dcp_size`` DCP groups that share the same KV.  Affects
    reader counts and writer selection for MLA."""

    pcp_size: int = 1
    """Prefill context parallel size.  Included in world_size but does
    not affect KV cache sharding (same as TP for LMCache purposes)."""

    @property
    def ranks_per_node(self) -> int:
        """Number of vLLM ranks assigned to each LMCache server.

        Server blocks are contiguous in rank space (see the connector's
        ``local_server_url = server_urls[rank // ranks_per_node]``):
        server 0 → ranks ``[0, ranks_per_node)``, server 1 →
        ``[ranks_per_node, 2*ranks_per_node)``, etc.
        """
        return self.vllm_world_size // self.n_servers

    @property
    def kv_world_size(self) -> int:
        """Number of pieces a single token chunk's KV cache is split into
        on the LMCache server storage."""
        if self.use_mla:
            if self.tp_size == 0:
                return 0
            return self.vllm_world_size // self.tp_size
        if self.n_servers == 0:
            return 0
        return self.vllm_world_size // self.n_servers

    @property
    def kv_worker_id(self) -> int:
        """Index of the piece of a single token chunk's KV cache
        that the current worker is responsible for,
        in ``[0, kv_world_size)``."""
        if self.use_mla:
            if self.tp_size == 0:
                return 0
            return self.vllm_worker_id // self.tp_size
        rpn = self.ranks_per_node
        if rpn == 0:
            return 0
        return self.vllm_worker_id % rpn

    @property
    def kv_tp_size(self) -> int:
        """Number of readers that will retrieve the same KV object from a
        single LMCache server.

        Non-MLA: each TP worker owns a distinct shard, so exactly 1 reader
        per object → ``tp_size // n_servers`` (shards per server).

        MLA: all TP workers within a pipeline stage share one KV object.
        When ``ranks_per_node >= tp_size`` (the common case: each server
        holds one or more complete PP stages), every ``tp_size`` rank in
        a stage lands on the same server and reads the same object, so
        the reader count is ``tp_size``.  When ``ranks_per_node <
        tp_size`` (more servers than pipeline stages), the TP group is
        split across servers and each server gets ``ranks_per_node``
        readers.

        Under DCP (decode context parallel), each TP group is split
        into ``tp_size // dcp_size`` DCP groups that all read the same
        MLA KV object.  The effective reader count is multiplied by
        ``dcp_size``: ``min(tp_size * dcp_size, ranks_per_node)``.

        The old formula ``tp_size // n_servers`` was correct only when
        server blocks split TP groups evenly (PP=1).  Under PP > 1 the
        server blocks align with PP stages, not TP groups, causing
        under-locking (e.g. TP=4 PP=2 ns=2: locked 2, actual readers 4).
        """
        if self.use_mla:
            effective_tp = self.tp_size * self.dcp_size
            return min(effective_tp, self.ranks_per_node)
        return self.tp_size // self.n_servers

    @property
    def is_kv_writer(self) -> bool:
        """Whether this rank is responsible for storing KV.

        Non-MLA: every rank writes its own distinct shard.

        MLA: all TP workers in a (server, pipeline-stage) pair share one
        KV object, so exactly one rank per pair should write.  The
        writer is the first TP-local rank within the server block:
        ``(rank % ranks_per_node) % tp_size == 0``.

        This yields exactly one writer per (server, stage) for all
        practical configurations where ``ranks_per_node >= 1`` and
        ``ranks_per_node % tp_size == 0`` (i.e. server blocks are
        TP-aligned, which holds for every standard TP×PP×ns layout
        because vLLM assigns ranks TP-inner/PP-outer and the connector
        splits them into contiguous blocks).  The old formula
        ``rank % (tp_size // n_servers) == 0`` selected multiple
        writers per shard when PP > 1 (double-stores).
        """
        if not self.use_mla:
            return True
        rpn = self.ranks_per_node
        if rpn == 0:
            return True
        # Under DCP, the effective TP group is tp_size * dcp_size.
        # The writer is the first rank in each (server, stage, dcp_group).
        effective_tp = self.tp_size if self.dcp_size <= 1 else self.tp_size
        # When dcp_size > 1, dcp workers share KV, so the writer condition
        # uses tp_size (not tp_size * dcp_size) because only one DCP rank
        # per TP rank writes.
        return (self.vllm_worker_id % rpn) % effective_tp == 0

    @property
    def dp_server_offset(self) -> int:
        """Server URL index offset for this DP replica.

        When ``dp_size > 1``, each DP replica gets
        ``n_servers // dp_size`` servers.  The offset is
        ``dp_rank * (n_servers // dp_size)`` so replicas don't collide.
        """
        if self.dp_size <= 1:
            return 0
        servers_per_dp = self.n_servers // self.dp_size
        return self.dp_rank * servers_per_dp

    @property
    def dp_rank_value(self) -> int:
        """The DP rank to embed in cache keys (isolates DP replicas)."""
        return self.dp_rank


@dataclass
class LoadStoreOp:
    block_ids: list[int]
    """Block ids for the load/store operation"""

    token_ids: list[int] | None = None
    """Token IDs for the load/store operation (token mode)"""

    block_hashes: list[bytes] | None = None
    """Block hashes for the load/store operation (hash mode)"""

    start: int = 0
    """Start token index (token mode only)"""

    end: int = 0
    """End token index (token mode only)"""

    def __len__(self) -> int:
        return len(self.block_ids)


StoreResult = bool
RetrieveResult = list[bool]
LookupResult = int


class LMCacheMPSchedulerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
    ):
        """
        Args:
            server_url: The server URL for the LMCache message queue
            context: The ZMQ context

            model_name: The model name used for LMCache keys
            vllm_block_size: The block size used in vLLM
            parallel_strategy:
                The parallel strategy, which includes `use_mla`,
                `world_size`, `worker_id` and so on
        """
        self.mq_client = MessageQueueClient(server_url, context)

        # Lookup state tracking (mirrors the LMCache-tree scheduler
        # adapter): a submitted lookup is tracked in ``_pending_lookups``
        # until its prefetch result is observed via
        # ``QUERY_PREFETCH_STATUS``; the aggregated token count is then
        # cached in ``_finished_lookup_results`` so repeated polls return
        # the same value (exactly-once).  ``_lookup_params`` remembers the
        # token_ids + cache_salt so ``free_lookup_locks`` can rebuild the
        # key without the caller re-passing them.
        self._pending_lookups: set[str] = set()
        self._finished_lookup_results: dict[str, int] = {}
        self._lookup_params: dict[str, tuple[list[int], str]] = {}

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        self.chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert self.chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = self.chunk_size // vllm_block_size

    @property
    def world_size(self) -> int:
        """The world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """The worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def tp_size(self) -> int:
        """Per-server tensor-parallel reader count for MLA multi-reader
        locking.  Mirrors ``ParallelStrategy.kv_tp_size`` so the value
        passed in LOOKUP / FREE_LOOKUP_LOCKS payloads matches the
        LMCache-tree adapter (the server uses it as the reader count in
        ``compute_extra_count``).
        """
        return self.parallel_strategy.kv_tp_size

    @_lmcache_nvtx_annotate
    def maybe_submit_lookup_request(
        self,
        request_id: str,
        token_ids: list[int],
        cache_salt: str = "",
    ) -> None:
        """Submit a new lookup request to LMCache if there is no ongoing one.

        Token-based only (the hash-mode path was removed upstream along
        with ``IPCCacheServerKey.chunk_hash``).  Truncates to a
        chunk-aligned length, builds a single no-worker-id key, and sends
        ``LOOKUP [key, tp_size]`` to the server — the payload shape the
        post-#4082 server expects (it reads ``tp_size`` as the reader
        count for ``compute_extra_count``).

        Args:
            request_id: The ID of the lookup request. The same ID
                indicates it's from the same request.
            token_ids: Token IDs to lookup from LMCache.
            cache_salt: Per-user isolation salt. Requests with different
                cache_salt values produce separate cache entries.

        Notes:
            Side effect: submits a LOOKUP, which "locks" the matched KV
            cache chunks for later retrieve.  The result is polled via
            :meth:`check_lookup_result` (which sends
            ``QUERY_PREFETCH_STATUS`` — the LOOKUP call itself returns
            ``None`` on the new server).
        """
        if request_id in self._pending_lookups:
            # Skip if there is already a lookup request
            return

        aligned_end = (len(token_ids) // self.chunk_size) * self.chunk_size
        if aligned_end == 0:
            return

        key = self._create_key(
            token_ids,
            start=0,
            end=aligned_end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()

        send_lmcache_request(
            self.mq_client,
            RequestType.LOOKUP,
            [key, self.tp_size],
        )
        self._pending_lookups.add(request_id)
        self._lookup_params[request_id] = (token_ids, cache_salt)

    @_lmcache_nvtx_annotate
    def check_lookup_result(self, request_id: str) -> int | None:
        """Check the result of a previously submitted lookup request.

        Sends a ``QUERY_PREFETCH_STATUS`` request and blocks until the
        server responds.  Returns the matched token count when the
        prefetch is complete, or ``None`` if still in progress.  The
        aggregated count is cached so repeated calls are idempotent
        (the server pops the job after the first successful poll).

        Args:
            request_id: The ID of the lookup request submitted in
                :meth:`maybe_submit_lookup_request`.

        Returns:
            The total number of matched tokens (prefix matching), or
            ``None`` if the lookup is not finished yet, or ``0`` if no
            lookup was ever submitted (e.g. empty prompt).
        """
        if request_id not in self._pending_lookups:
            # No job submitted — return cached aggregate if any, else 0.
            return self._finished_lookup_results.get(request_id, 0)

        if request_id in self._finished_lookup_results:
            # Already aggregated; return the cached value.
            return self._finished_lookup_results[request_id]

        future = send_lmcache_request(
            self.mq_client,
            RequestType.QUERY_PREFETCH_STATUS,
            [request_id],
        )
        result = future.result()
        if result is None:
            return None

        num_chunks = int(result)
        token_count = num_chunks * self.chunk_size
        self._finished_lookup_results[request_id] = token_count
        return token_count

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def cleanup_lookup_result(self, request_id: str) -> None:
        """Clean up lookup state for a finished request to prevent leaks.

        Args:
            request_id: The ID of the finished request.
        """
        self._pending_lookups.discard(request_id)
        self._finished_lookup_results.pop(request_id, None)
        self._lookup_params.pop(request_id, None)

    def free_lookup_locks(
        self,
        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> None:
        """Release read locks acquired during lookup without a full retrieve.

        Use this when some chunks matched by lookup overlap with blocks
        vLLM has already computed (so they will never be retrieved), or
        when a request is aborted after lookup but before retrieve.

        When ``start``/``end`` are not chunk-aligned, the chunk containing
        the ``start`` boundary is freed but not the ``end`` boundary —
        it is the caller's responsibility to align boundaries.

        Mirrors the LMCache-tree scheduler adapter: builds a
        no-worker-id key over ``[start, end)`` and sends
        ``FREE_LOOKUP_LOCKS [key, tp_size]``.
        """
        key = self._create_key(
            token_ids,
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        ).no_worker_id_version()
        send_lmcache_request(
            self.mq_client,
            RequestType.FREE_LOOKUP_LOCKS,
            [key, self.tp_size],
        )

    def end_session(self, request_id: str) -> None:
        """
        Notify LMCache server to remove the session for a finished request.
        Args:
            request_id: The ID of the finished request.
        """
        send_lmcache_request(
            self.mq_client,
            RequestType.END_SESSION,
            [request_id],
        )

    # Helper functions
    def _create_key(
        self,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str | None = None,
        cache_salt: str = "",
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Passes the same field set as the LMCache-tree adapter's
        ``_create_key``: ``cache_salt`` (per-user isolation),
        ``dp_rank`` (DP-replica key isolation), and ``device_vendor``
        (server-side vendor routing).  Omitting any of these would
        diverge from the real adapter's wire format.
        """
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=None,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id if request_id is not None else "",
            cache_salt=cache_salt,
            use_mla=self.parallel_strategy.use_mla,
            dp_rank=self.parallel_strategy.dp_rank_value,
            device_vendor=_device_vendor(),
        )

    def _create_hash_key(
        self, chunk_hash: bytes, request_id: str | None = None
    ) -> IPCCacheServerKey:
        """Create a hash-mode IPC cache engine key.

        .. deprecated::
            The current ``IPCCacheServerKey`` API is token-based (no
            ``chunk_hash`` field).  Hash-mode keys are no longer
            supported.  This method is retained for API compatibility
            with callers that haven't been migrated to the token-based
            ``_create_key`` path.  When the lmcache package is installed
            (the normal case), the LMCache-tree adapter is used instead
            of this vendored fallback, so this method is never reached.
        """
        raise NotImplementedError(
            "Hash-mode IPCCacheServerKey is no longer supported. "
            "Use _create_key (token-based) instead, or install the "
            "lmcache package to use the LMCache-tree adapter."
        )


class LMCacheMPWorkerAdapter:
    def __init__(
        self,
        server_url: str,
        context: zmq.Context,
        model_name: str,
        vllm_block_size: int,
        parallel_strategy: ParallelStrategy,
    ):
        self.mq_client = MessageQueueClient(server_url, context)

        # Instance id for GPU worker
        self.instance_id = os.getpid()

        # Registered kv caches from vLLM
        self.kv_caches: dict[str, torch.Tensor] = {}

        # Request futures
        # request_id -> (future, other merged requests)
        self.store_futures: dict[
            str, tuple[MessagingFuture[StoreResult], list[str]]
        ] = {}
        self.retrieve_futures: dict[
            str, tuple[MessagingFuture[RetrieveResult], list[str]]
        ] = {}

        # The store requests that have finished execution in LMCache
        self.finished_stores: set[str] = set()
        # The finished request ids that are passed via vLLM and also
        # have corresponding store requests submitted to LMCache before
        self.previously_finished: set[str] = set()

        # Block IDs that failed due to retrieve timeout/error.  Surfaced
        # to the connector via get_block_ids_with_load_errors() so vLLM
        # recomputes them.  Mirrors the LMCache-tree worker adapter.
        self.error_block_ids: set[int] = set()

        self.model_name = model_name
        self.parallel_strategy = parallel_strategy

        # Read chunk size from lmcache
        chunk_size = get_lmcache_chunk_size(self.mq_client)
        assert chunk_size % vllm_block_size == 0, (
            "LMCache chunk size should be a multiple of vLLM block size"
        )
        self.blocks_in_chunk = chunk_size // vllm_block_size

    @property
    def world_size(self) -> int:
        """The world size."""
        return self.parallel_strategy.kv_world_size

    @property
    def worker_id(self) -> int:
        """The worker id."""
        return self.parallel_strategy.kv_worker_id

    @property
    def use_mla(self) -> bool:
        """Whether to use MLA."""
        return self.parallel_strategy.use_mla

    @property
    def is_kv_writer(self) -> bool:
        """Whether this rank is responsible for storing KV.

        Delegates to ``ParallelStrategy.is_kv_writer`` so the vendored
        fallback matches the LMCache-tree adapter: exactly one writer per
        (server, pipeline-stage) for MLA, every rank for non-MLA.
        Replaces the old ``is_first_rank_of_pp_group`` which used the
        removed ``actual_worker_id`` field and selected multiple writers
        per shard under PP > 1.
        """
        return self.parallel_strategy.is_kv_writer

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """
        Register the kv caches with LMCache server

        Args:
            kv_caches: A dict of kv caches to register. The keys are the
                layer names and the values are the corresponding tensors.
        """
        # Register kv cache and send the request
        self.kv_caches = kv_caches
        logger.info("Registering kv caches")
        future = send_lmcache_request(
            self.mq_client,
            RequestType.REGISTER_KV_CACHE,
            [self.instance_id, wrap_kv_caches(kv_caches)],
        )
        future.result()

    @_lmcache_nvtx_annotate
    def submit_store_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache store request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the store operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        if op.block_hashes is not None:
            # Hash mode
            chunk_hashes = list(
                striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode
            assert op.token_ids is not None
            keys = [
                self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
            ]
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.store_futures[request_id] = (future, [])

    @_lmcache_nvtx_annotate
    def submit_retrieve_request(
        self, request_id: str, op: LoadStoreOp, event: torch.cuda.Event
    ):
        """
        Submit a KV cache retrieve request to LMCache

        Args:
            request_id: The ID of the request
            op: The LoadStoreOp describing the retrieve operation.
            event: The CUDA event that is recorded after the current
                model inference step
        """
        if op.block_hashes is not None:
            # Hash mode
            chunk_hashes = list(
                striding_block_hashes(op.block_hashes, self.blocks_in_chunk)
            )
            keys = [
                self._create_hash_key(ch, request_id=request_id) for ch in chunk_hashes
            ]
        else:
            # Token mode
            assert op.token_ids is not None
            keys = [
                self._create_key(op.token_ids, op.start, op.end, request_id=request_id)
            ]
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [keys, self.instance_id, op.block_ids, event.ipc_handle()],
        ).to_cuda_future()
        self.retrieve_futures[request_id] = (future, [])

    @_lmcache_nvtx_annotate
    def batched_submit_store_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched store request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the store operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. When
                ``None`` (older callers) all requests use the empty salt.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        all_keys: list[IPCCacheServerKey] = []
        block_ids: list[int] = []
        for request_id, op, cache_salt in zip(
            request_ids, ops, cache_salts, strict=False
        ):
            # Token-based only (hash mode was removed upstream).
            assert op.token_ids is not None
            all_keys.append(
                self._create_key(
                    op.token_ids,
                    op.start,
                    op.end,
                    request_id=request_id,
                    cache_salt=cache_salt,
                )
            )
            block_ids.extend(op.block_ids)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.STORE,
            [
                all_keys,
                self.instance_id,
                block_ids,
                event.ipc_handle(),
            ],
        ).to_cuda_future()
        self.store_futures[request_ids[0]] = (future, list(request_ids[1:]))

    @_lmcache_nvtx_annotate
    def batched_submit_retrieve_requests(
        self,
        request_ids: list[str],
        ops: list[LoadStoreOp],
        event: torch.cuda.Event,
        cache_salts: list[str] | None = None,
    ):
        """
        Submit a batched retrieve request to LMCache

        Args:
            request_ids: The IDs of the requests
            ops: The LoadStoreOps describing the retrieve operations. Should have
                the same length as request_ids
            event: The CUDA event that is recorded after the current
                model inference step
            cache_salts: Per-user isolation salts, one per request. When
                ``None`` (older callers) all requests use the empty salt.
        """
        if cache_salts is None:
            cache_salts = [""] * len(request_ids)
        all_keys: list[IPCCacheServerKey] = []
        block_ids: list[int] = []
        for request_id, op, cache_salt in zip(
            request_ids, ops, cache_salts, strict=False
        ):
            # Token-based only (hash mode was removed upstream).
            assert op.token_ids is not None
            all_keys.append(
                self._create_key(
                    op.token_ids,
                    op.start,
                    op.end,
                    request_id=request_id,
                    cache_salt=cache_salt,
                )
            )
            block_ids.extend(op.block_ids)
        future = send_lmcache_request(
            self.mq_client,
            RequestType.RETRIEVE,
            [
                all_keys,
                self.instance_id,
                block_ids,
                event.ipc_handle(),
            ],
        ).to_cuda_future()
        self.retrieve_futures[request_ids[0]] = (future, list(request_ids[1:]))

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids_from_engine: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Check and get the finished store and retrieve requests.

        Args:
            finished_req_ids_from_engine: the set of request ids that are
                reported as finished from the vLLM engine side.

        Returns:
            A tuple of two sets:
            - The first set contains the finished store request ids. The returned
                store request ids MUST be seen before in the
                `finished_req_ids_from_engine`.
            - The second set contains the finished retrieve request ids.

        Notes:
            When enabling async scheduling in vLLM, the same request ID may appear
            multiple times in `finished_req_ids_from_engine`. The adapter should
            take care of deduplicating the request IDs and only return the request
            IDs that have not been returned before.
        """
        finished_stores = set()
        finished_retrieves = set()
        for request_id, (s_future, other_reqs) in self.store_futures.items():
            if not s_future.query():
                continue

            s_result = s_future.result()
            finished_stores.add(request_id)
            finished_stores.update(other_reqs)

            if not s_result:
                # TODO: add error handling here
                logger.error(
                    "Something went wrong when processing the "
                    "store request for request_id=%s",
                    request_id,
                )

        for request_id, (r_future, other_reqs) in self.retrieve_futures.items():
            if not r_future.query():
                continue

            r_result = r_future.result()
            finished_retrieves.add(request_id)
            finished_retrieves.update(other_reqs)

            if not all(r_result):
                # TODO: add error handing here
                logger.error(
                    "Something went wrong when processing the "
                    "retrieve request for request_id=%s, result=%s",
                    request_id,
                    r_result,
                )

        # Remove the finished requests from the tracking dicts
        for request_id in finished_stores:
            self.store_futures.pop(request_id, None)
        for request_id in finished_retrieves:
            self.retrieve_futures.pop(request_id, None)

        # Update the internal states
        self.finished_stores.update(finished_stores)

        ret_stores = set()
        for req_id in finished_req_ids_from_engine:
            if req_id in self.finished_stores or req_id in self.store_futures:
                self.previously_finished.add(req_id)
            else:
                ret_stores.add(req_id)

        # Calculate the final finished stores
        ret_stores.update(self._update_and_get_finished_store())

        return ret_stores, finished_retrieves

    def num_blocks_per_chunk(self) -> int:
        """
        Returns:
            The number of vllm blocks in a LMCache data chunk
        """
        return self.blocks_in_chunk

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return the set of vLLM block IDs whose retrieve failed.

        The connector calls this each step so vLLM recomputes the
        affected blocks instead of using stale/empty KV.  Once read,
        the set is cleared so each failure is reported exactly once.
        Mirrors the LMCache-tree worker adapter.
        """
        if not self.error_block_ids:
            return set()
        errors = set(self.error_block_ids)
        self.error_block_ids.clear()
        return errors

    def shutdown(self):
        """
        Shutdown the LMCache MP worker adapter
        """
        logger.info("Unregistering kv caches")
        send_lmcache_request(
            self.mq_client, RequestType.UNREGISTER_KV_CACHE, [self.instance_id]
        ).result()

        self.mq_client.close()

    # Helper functions
    def _update_and_get_finished_store(
        self,
    ) -> set[str]:
        """Converge the internal states about finished stores
        and returns the 'safe finished store request ids' back
        """
        safe_finished_s = self.finished_stores.intersection(self.previously_finished)
        self.finished_stores.difference_update(self.previously_finished)
        self.previously_finished.difference_update(safe_finished_s)

        return safe_finished_s

    def _create_key(
        self,
        token_ids: list[int],
        start: int = 0,
        end: int = 0,
        request_id: str | None = None,
        cache_salt: str = "",
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Passes the same field set as the LMCache-tree worker adapter's
        ``_create_key``: ``cache_salt`` (per-user isolation), ``dp_rank``
        (DP-replica key isolation), and ``device_vendor`` (server-side
        vendor routing).
        """
        return IPCCacheServerKey(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id if request_id is not None else "",
            cache_salt=cache_salt,
            use_mla=self.parallel_strategy.use_mla,
            dp_rank=self.parallel_strategy.dp_rank_value,
            device_vendor=_device_vendor(),
        )

    def _create_hash_key(
        self, chunk_hash: bytes, request_id: str | None = None
    ) -> IPCCacheServerKey:
        """Create a hash-mode IPC cache engine key.

        .. deprecated::
            Hash-mode keys are no longer supported by the current
            ``IPCCacheServerKey`` API (token-based only).  See the
            scheduler adapter's ``_create_hash_key`` for details.
        """
        raise NotImplementedError(
            "Hash-mode IPCCacheServerKey is no longer supported. "
            "Use _create_key (token-based) instead."
        )
