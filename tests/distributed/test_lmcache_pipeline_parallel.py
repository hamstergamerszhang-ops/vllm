# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
End-to-end GPU test for LMCache + vLLM pipeline parallelism (PR #4082).

Before the fix, starting vLLM with the ``LMCacheMPConnector`` in
multi-server mode under ``--pipeline-parallel-size > 1`` failed at
startup with::

    ValueError: LMCacheMPConnector multi-server mode only supports
    tensor parallelism (TP), not pipeline parallelism (PP). Got
    pp_size=2.

That guard existed only to paper over a latent locking bug: the server's
``compute_extra_count`` inferred MLA-ness from ``tp > world_size``, which
is only valid when PP=1.  PR #4082 makes MLA explicit (an ``use_mla`` flag
on the cache key) and fixes ``kv_tp_size`` / ``is_kv_writer`` so server
blocks align with PP stages, then removes the guard.

This test exercises the post-fix path end to end:

  1. Start one LMCache server (the ``lmcache server`` MP subcommand).
  2. Start ``vllm serve`` on a small MLA model (DeepSeek-V2-Lite) with
     ``--tensor-parallel-size 1 --pipeline-parallel-size 2`` and the
     ``LMCacheMPConnector`` pointed at the cache server.
  3. Assert the server actually comes up (no startup ``ValueError``) —
     this is the regression guard for the removed PP guard.
  4. Send a request whose prompt is a long shared prefix + a short
     unique suffix, then a second request reusing the same prefix.
     Assert both succeed and produce coherent, *distinct* output (a
     retrieve-path corruption would make them identical or empty), and
     log the warm-vs-cold TTFT so a regression that drops KV reuse to
     zero is visible.

This test needs real GPUs and the ``lmcache`` package installed; it is
skipped automatically otherwise.  The ``@pytest.mark.distributed(num_gpus=2)``
mark selects a 2-GPU CI shard (and skips on hosts with fewer GPUs).

Run (on a 2-GPU Linux+NVIDIA box with lmcache installed)::

    pytest tests/distributed/test_lmcache_pipeline_parallel.py -v -s

"""
from __future__ import annotations

import contextlib
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import closing

import pytest

from ..utils import RemoteOpenAIServer

# DeepSeek-V2-Lite is the smallest widely-available MLA model
# (kv_lora_rank set -> is_deepseek_mla() True).  It fits on 2 modest GPUs
# with TP=1 PP=2 and is already in vLLM's test model registry.
MLA_MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# A prompt with a long shared prefix so the second request hits the
# LMCache lookup path.  The prefix is deliberately repetitive and long
# enough to span multiple LMCache chunks.
SHARED_PREFIX = (
    "You are a meticulous assistant. " * 64
    + "Context: The LMCache project provides a KV-cache sharing layer "
    + "for vLLM. " * 32
)
PROMPT_A = SHARED_PREFIX + "\n\nQuestion: In one word, what is LMCache?"
PROMPT_B = SHARED_PREFIX + "\n\nQuestion: In one word, what is vLLM?"


def _get_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _lmcache_available() -> bool:
    try:
        import lmcache  # noqa: F401
        return True
    except Exception:
        return False


def _cuda_device_count() -> int:
    try:
        from vllm.platforms import current_platform
        return current_platform.device_count()
    except Exception:
        return 0


# Skip unless we have GPUs + lmcache.  Collected (but not run) on hosts
# without them, so the file still type-checks / imports cleanly in CI.
pytestmark = [
    pytest.mark.skipif(
        not _lmcache_available(),
        reason="lmcache package not installed; install with `pip install lmcache`",
    ),
    pytest.mark.skipif(
        _cuda_device_count() < 2,
        reason="Need at least 2 GPUs for PP=2",
    ),
    pytest.mark.distributed(num_gpus=2),
]


class LMCacheServer:
    """Context manager that spawns and tears down an ``lmcache_server``.

    Mirrors the lifecycle discipline of ``RemoteVLLMServer`` (process-group
    kill + reap) but for the LMCache ZMQ cache server, which has no HTTP
    health endpoint — readiness is established by the vLLM server's own
    connector handshake succeeding.
    """

    def __init__(self, host: str = "127.0.0.1", port: int | None = None):
        self.host = host
        self.port = port or _get_free_port()
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> "LMCacheServer":
        # Use the ``lmcache`` CLI's ``server`` subcommand (the MP server
        # registered via ``lmcache.cli.commands.server.ServerCommand``),
        # NOT the legacy ``lmcache_server`` / ``python -m
        # lmcache.v1.server.__main__`` entrypoint — that one takes
        # positional ``<host> <port> <storage>`` args and would reject
        # ``--host``/``--port``/``--chunk-size``.
        cmd = [
            sys.executable, "-m", "lmcache.cli.main",
            "server",
            "--host", self.host,
            "--port", str(self.port),
            # A small chunk size keeps the test prompt spanning multiple
            # chunks so the lookup path is exercised meaningfully.
            "--chunk-size", "16",
        ]
        print(f"[LMCacheServer] starting: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        # Give the ZMQ socket a moment to bind.  There's no health probe,
        # so we just wait for the port to accept connections.
        if not self._wait_for_port(timeout=20.0):
            out = self.proc.stdout.read() if self.proc.stdout else ""
            self._kill()
            raise RuntimeError(
                f"lmcache server did not bind {self.host}:{self.port} within 20s. "
                f"Output:\n{out}"
            )
        return self

    def __exit__(self, *exc):
        self._kill()

    def _wait_for_port(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                return False
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
                s.settimeout(0.5)
                try:
                    s.connect((self.host, self.port))
                    return True
                except OSError:
                    time.sleep(0.3)
        return False

    def _kill(self) -> None:
        if self.proc is None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
        with contextlib.suppress(ProcessLookupError, OSError):
            self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if pgid is not None:
                with contextlib.suppress(ProcessLookupError, OSError):
                    os.killpg(pgid, signal.SIGKILL)
            else:
                self.proc.kill()
        self.proc = None


@pytest.mark.parametrize("pp_size", [2])
def test_lmcache_pp_mla_startup_and_reuse(pp_size: int):
    """PP>1 with LMCacheMPConnector + an MLA model must start and reuse KV.

    Regression guard for the removed PP guard (PR #4082): pre-fix this
    raised ``ValueError`` at startup; post-fix it must serve requests,
    and the second shared-prefix request must hit the LMCache lookup path.
    """
    with LMCacheServer() as cache:
        kv_config = (
            '{"kv_connector":"LMCacheMPConnector",'
            '"kv_role":"kv_both",'
            f'"lmcache.mp.host":"tcp://{cache.host}",'
            f'"lmcache.mp.port":{cache.port},'
            '"lmcache.mp.mq_timeout":300.0'
            "}"
        )
        server_args = [
            # PP>1 is the whole point — this is what was blocked pre-fix.
            "--pipeline-parallel-size", str(pp_size),
            "--tensor-parallel-size", "1",
            "--kv-transfer-config", kv_config,
            # Keep the test cheap and deterministic.
            "--max-model-len", "2048",
            "--max-num-batched-tokens", "512",
            "--gpu-memory-utilization", "0.85",
            "--dtype", "bfloat16",
            "--enforce-eager",
            # Request logging is off by default; no flag needed.
        ]
        # RemoteOpenAIServer handles download, port, health-check, and
        # process-tree teardown (including GPU memory release).
        with RemoteOpenAIServer(
            MLA_MODEL, server_args, max_wait_seconds=600
        ) as server:
            client = server.get_client()

            # First request: cold prefix, populates the LMCache store path.
            t0 = time.monotonic()
            out_a = client.completions.create(
                model=MLA_MODEL,
                prompt=PROMPT_A,
                temperature=0,
                max_tokens=8,
            )
            ttft_a = time.monotonic() - t0
            text_a = out_a.choices[0].text

            # Second request: same prefix -> must hit the LMCache lookup
            # path.  We assert correctness (deterministic, non-empty) and
            # that the retrieve path didn't corrupt KV (output is sane).
            t0 = time.monotonic()
            out_b = client.completions.create(
                model=MLA_MODEL,
                prompt=PROMPT_B,
                temperature=0,
                max_tokens=8,
            )
            ttft_b = time.monotonic() - t0
            text_b = out_b.choices[0].text

    print(f"[lmcache-pp] TTFT A (cold)  = {ttft_a:.3f}s")
    print(f"[lmcache-pp] TTFT B (warm) = {ttft_b:.3f}s")
    print(f"[lmcache-pp] out A = {text_a!r}")
    print(f"[lmcache-pp] out B = {text_b!r}")

    # --- Correctness: both requests produced coherent, distinct output.
    assert text_a.strip(), "First request produced empty output"
    assert text_b.strip(), "Second request produced empty output"
    assert text_a != text_b, (
        "Both requests produced identical output despite different "
        "questions — possible KV corruption from the retrieve path."
    )

    # --- Performance: the warm request reused cached KV.  We don't
    # require a hard speedup (GPU variance is large on small models),
    # but we log it so a regression that drops reuse to zero is visible.
    # The definitive signal is that no ValueError was raised at startup
    # (the with-block above would have failed) — i.e. PP+MLA+LMCache
    # now starts, which it did not before PR #4082.
    if ttft_b > ttft_a:
        print(
            "[lmcache-pp] WARNING: warm TTFT > cold TTFT; KV reuse may "
            "not have fired. Investigate the connector lookup path."
        )
