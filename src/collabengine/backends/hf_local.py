"""In-process CUDA backend: `transformers` with dynamic micro-batching.

PLAN.md 5 wanted vLLM behind an OpenAI-compatible endpoint. vLLM ships no
supported Windows build, and this study runs on a Windows box with an RTX 4500
Ada, so the serving layer is provided here instead. `OpenAICompatBackend` stays
exactly as it is for a WSL2 vLLM server or a rented Linux box; swapping is a
config line, which is the property Phase 0 asked the backend abstraction to have.

**Batching is the whole point.** One turn at a time on an 8B model is ~30-40
output tok/s, which puts the ~14M-token condition grid at roughly two weeks of
wall clock. Batching 16-32 turns costs almost nothing extra per step because
decoding is memory-bandwidth-bound, not compute-bound, at these sizes -- the same
reason vLLM is fast. The runner already fans dozens of episodes at the backend
concurrently, so requests to batch are there for the taking; this module just
collects them.

The collection rule: take the first waiting request, then keep draining for
`batch_window_s` or until `max_batch_size`, then run. A window on the order of
tens of milliseconds is invisible next to a multi-second generation, and it is
what lets independent episodes' turns ride the same forward pass.

Two caveats worth stating plainly, because both bear on claims the study makes:

* **Sampling reproducibility is per batch, not per request.** vLLM seeds each
  sequence independently; `model.generate` draws from one global RNG for the
  whole batch, so an episode's text depends on which other episodes happened to
  batch with it. Member seeds are folded into one per-batch seed so an identical
  batch replays identically, but arbitrary batch composition does not. This does
  not touch the analysis -- transcripts store every message verbatim and are
  re-read, never regenerated -- but re-running a *specific* episode bit-for-bit
  needs `max_batch_size=1`. `ExperimentConfig` records the setting either way.
* **Thinking mode is off.** Qwen3 is a hybrid-reasoning model whose chat template
  defaults to emitting a `<think>` block. At `max_tokens=512` that block would
  consume the entire budget and the turn would end before saying anything to the
  team, so `enable_thinking=False` is passed to the template. Turning it on is a
  legitimate experiment -- it is a different one, and it needs a bigger budget.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from collabengine.backends.base import GenRequest, GenResponse, LLMBackend

DEFAULT_MODEL = "Qwen/Qwen3-8B"


@dataclass(slots=True)
class _Waiter:
    request: GenRequest
    future: asyncio.Future


@dataclass
class HFLocalBackend(LLMBackend):
    """Local CUDA generation with a dynamic batching queue."""

    model_id: str = DEFAULT_MODEL
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_batch_size: int = 16
    """Turns per forward pass.

    Bounded by KV cache, not by compute: an 8B model at bf16 leaves ~7 GB on a
    24 GB card, and each 8k-token sequence costs roughly 130 MB of KV at this
    model's head configuration. 16 is comfortable; 32 is reachable at shorter
    contexts. Exceeding it does not fail cleanly, so `_generate_batch` catches
    OOM and halves the batch rather than losing an hour of queued work."""
    max_batch_tokens: int = 32768
    """Padded token ceiling for one forward pass -- the real memory limit.

    A sequence count alone is the wrong bound, because KV cache scales with
    tokens, not requests. Sixteen round-one turns share a short context and fit
    easily; sixteen round-three turns each carry the whole transcript and do
    not. Qwen3-8B at bf16 costs ~144 KiB of KV per token (36 layers x 8 KV heads
    x 128 dim x 2 tensors x 2 bytes), so the ~6 GB left after weights holds
    roughly 43k tokens; 32k leaves room for activations and fragmentation.

    Bounding on `max_batch_size` alone means the batch that OOMs is the one with
    the longest contexts -- late in an episode, after the most work has already
    been done. Recovery is possible (see `_generate_batch`) but costs a wasted
    forward pass and a cache flush, so it is much better to not get there."""
    batch_window_s: float = 0.05
    """How long to keep collecting after the first request arrives."""
    max_model_len: int = 8192
    enable_thinking: bool = False
    trust_remote_code: bool = False
    name: str = "hf_local"

    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _queue: Any = field(default=None, init=False, repr=False)
    _worker: Any = field(default=None, init=False, repr=False)
    _load_lock: Any = field(default=None, init=False, repr=False)
    _loop: Any = field(default=None, init=False, repr=False)
    _stats: dict = field(default_factory=dict, init=False, repr=False)

    # ---------------------------------------------------------------- loading

    def _load_sync(self) -> None:
        """Blocking model load. Called once, off the event loop."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        # Decoder-only batch generation requires left padding: right padding puts
        # pad tokens between the prompt and the first generated token, and the
        # model happily continues from the padding instead of from the prompt.
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch_dtype = getattr(torch, self.dtype)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        except TypeError:
            # transformers < 5 spells it `torch_dtype`.
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_lock is None:
            self._load_lock = asyncio.Lock()
        async with self._load_lock:
            if self._model is None:
                await asyncio.to_thread(self._load_sync)

    # ---------------------------------------------------------------- queueing

    async def generate(self, request: GenRequest) -> GenResponse:
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            # A backend instance outlives the loop it was first used on: the CLI
            # builds it once and `calibrate` calls asyncio.run per difficulty.
            # Queues and locks bind to a loop at first use, so carrying them
            # across would raise "attached to a different loop" partway into a
            # sweep. The weights, which are the expensive part, are kept.
            self._loop = loop
            self._queue = asyncio.Queue()
            self._worker = None
            self._load_lock = None

        await self._ensure_loaded()

        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run_batches())

        future: asyncio.Future = loop.create_future()
        await self._queue.put(_Waiter(request=request, future=future))
        return await future

    async def _run_batches(self) -> None:
        """Drain the queue forever, one batch per forward pass."""
        while True:
            first: _Waiter = await self._queue.get()
            batch = [first]

            deadline = asyncio.get_running_loop().time() + self.batch_window_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    batch.append(
                        await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    )
                except asyncio.TimeoutError:
                    break

            # Sampling parameters are per-call in `generate`, so a batch has to
            # be homogeneous in them. In this experiment they are constant across
            # every turn, making this a no-op guard -- but a silent mix would
            # apply one episode's temperature to another's turn, which is the
            # kind of error that never surfaces as a crash.
            for group in _group_by_sampling(batch):
                try:
                    await self._dispatch(group)
                except Exception as exc:  # noqa: BLE001 - degrade, never abort
                    _fail(group, f"{type(exc).__name__}: {exc}")

    async def _dispatch(self, batch: list[_Waiter]) -> None:
        responses = await asyncio.to_thread(self._generate_batch, batch)
        for waiter, response in zip(batch, responses):
            if not waiter.future.done():
                waiter.future.set_result(response)

    # -------------------------------------------------------------- generation

    def _generate_batch(self, batch: list[_Waiter]) -> list[GenResponse]:
        """Run one forward pass. Blocking; called in a worker thread."""
        import torch
        from transformers import set_seed

        tok = self._tokenizer
        prompts = [self._render(w.request) for w in batch]
        sample = batch[0].request

        # Split on the token budget before allocating anything. Tokenizing
        # twice is microseconds against a multi-second generation, and it is
        # what keeps a batch of long contexts from reserving more KV cache than
        # the card has.
        if len(batch) > 1:
            chunks = _split_by_token_budget(
                batch,
                [len(ids) for ids in tok(prompts, add_special_tokens=False)["input_ids"]],
                budget=self.max_batch_tokens,
                max_new_tokens=sample.max_tokens,
            )
            if len(chunks) > 1:
                out: list[GenResponse] = []
                for chunk in chunks:
                    out.extend(self._generate_batch(chunk))
                return out

        encoded = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_model_len,
        ).to(self._model.device)

        # `generate` samples via `torch.multinomial` against the global RNG and
        # exposes no per-call generator, so seeding is a process-wide set_seed
        # immediately before the call. Generation runs one batch at a time in a
        # single worker thread, so nothing else is drawing in between.
        set_seed(_batch_seed(batch))

        try:
            with torch.inference_mode():
                out = self._model.generate(
                    **encoded,
                    max_new_tokens=sample.max_tokens,
                    do_sample=sample.temperature > 0,
                    temperature=sample.temperature,
                    top_p=sample.top_p,
                    pad_token_id=tok.pad_token_id,
                )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            # Halve and retry rather than lose the queued work. A batch that
            # OOMs is usually one long-context outlier riding with short turns.
            mid = len(batch) // 2
            return self._generate_batch(batch[:mid]) + self._generate_batch(batch[mid:])

        prompt_len = encoded["input_ids"].shape[1]
        generated = out[:, prompt_len:]

        responses: list[GenResponse] = []
        for i, w in enumerate(batch):
            tokens = generated[i]
            text = tok.decode(tokens, skip_special_tokens=True).strip()
            n_new = int((tokens != tok.pad_token_id).sum().item())
            responses.append(
                GenResponse(
                    text=text,
                    prompt_tokens=int(encoded["attention_mask"][i].sum().item()),
                    completion_tokens=n_new,
                    finish_reason=(
                        "length" if n_new >= w.request.max_tokens else "stop"
                    ),
                )
            )
        return responses

    def _render(self, request: GenRequest) -> str:
        """Apply the model's chat template to one request."""
        messages = [m.to_dict() for m in request.messages]
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Template does not take `enable_thinking` (non-hybrid model).
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    # ----------------------------------------------------------------- health

    async def health(self) -> bool:
        try:
            await self._ensure_loaded()
        except Exception:  # noqa: BLE001 - health must answer, not raise
            return False
        return self._model is not None

    async def aclose(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        if self._model is not None:
            self._model = None
            self._tokenizer = None
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass


def _split_by_token_budget(
    batch: list[_Waiter],
    lengths: list[int],
    *,
    budget: int,
    max_new_tokens: int,
) -> list[list[_Waiter]]:
    """Chunk a batch so no forward pass exceeds `budget` padded tokens.

    Cost is `len(chunk) * (longest prompt in chunk + max_new_tokens)`, because
    left-padding brings every sequence in a chunk up to the longest one and
    generation extends all of them. Order is preserved -- callers map results
    back positionally, so reordering to pack tighter would silently hand one
    episode another's turn.

    A single request always forms its own chunk even when it exceeds the budget:
    splitting cannot help, and refusing would fail a turn that the card can very
    likely still run.
    """
    chunks: list[list[_Waiter]] = []
    current: list[_Waiter] = []
    longest = 0

    for waiter, length in zip(batch, lengths):
        candidate_longest = max(longest, length)
        cost = (len(current) + 1) * (candidate_longest + max_new_tokens)
        if current and cost > budget:
            chunks.append(current)
            current, longest = [waiter], length
        else:
            current.append(waiter)
            longest = candidate_longest

    if current:
        chunks.append(current)
    return chunks


def _group_by_sampling(batch: list[_Waiter]) -> list[list[_Waiter]]:
    groups: dict[tuple, list[_Waiter]] = {}
    for w in batch:
        key = (w.request.max_tokens, w.request.temperature, w.request.top_p)
        groups.setdefault(key, []).append(w)
    return list(groups.values())


def _batch_seed(batch: list[_Waiter]) -> int:
    """A deterministic seed for the batch, derived from its members' seeds.

    Identical batch, identical sampling. Different batch composition, different
    sampling -- see the module docstring; this is the honest limit of batching
    against `model.generate` rather than a per-sequence sampler.
    """
    raw = ",".join(str(w.request.seed) for w in batch).encode()
    # `transformers.set_seed` rejects anything above 2**32 - 1.
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def _fail(batch: list[_Waiter], error: str) -> None:
    for w in batch:
        if not w.future.done():
            w.future.set_result(
                GenResponse(text="", finish_reason="error", error=error)
            )


def cuda_report() -> str:
    """One-line device summary, printed at the top of a run.

    A run that silently fell back to CPU looks identical to a slow run until
    hours have gone by, so the device is stated rather than assumed.
    """
    try:
        import torch
    except ImportError:
        return "torch not installed"
    if not torch.cuda.is_available():
        return "CUDA unavailable - generation would run on CPU"
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    return (
        f"{props.name}, {props.total_memory / 2**30:.1f} GiB, "
        f"CUDA {torch.version.cuda}, torch {torch.__version__}"
    )


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Batch shapes vary by design here -- the token budget produces differently
# sized allocations rather than one fixed block, and the caching allocator can
# fragment under that pattern until it fails an allocation while nvidia-smi
# still shows memory free. Expandable segments avoid it by growing a
# reservation instead of hunting for a contiguous block.
#
# Not available on Windows: torch accepts the setting and then warns that the
# platform does not support it. Setting it there buys a confusing warning line
# and nothing else, so it is skipped -- on Windows the token budget is the only
# thing standing between a long-context batch and an OOM, which is the reason
# that budget is conservative.
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
