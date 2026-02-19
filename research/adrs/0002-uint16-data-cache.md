# 0002. Use uint16 for Tokenized Data Cache

**Date:** 2026-02-18
**Status:** Accepted

## Context

The `TinyStoriesDataset` in `DataPipeline.py` tokenizes the full TinyStories
corpus, concatenates all tokens, chunks them into fixed-length sequences, and
saves the result to disk as a `.pt` file (e.g., `train_ctx256.pt`). This is a
one-time operation — subsequent runs load the cache directly and skip
tokenization entirely.

The tokenization + chunking step is the slowest part of initializing a new
training job on a remote GPU (Vast.ai). On a typical instance it takes several
minutes to download TinyStories from HuggingFace, tokenize ~2.1M stories with
GPT-2 BPE, and chunk the result. This cost is paid on every new instance
because the cache is local to the ephemeral GPU filesystem.

The obvious fix is to pre-build the cache on the dashboard droplet and SCP it
to each new GPU instance before training starts. However, the cache was stored
as `int64` tensors:

- 1,836,079 chunks x 257 tokens/chunk x 8 bytes = **~3.5 GB**

That is large enough to make SCP transfers slow and to consume meaningful
storage on the droplet. Since GPT-2's vocabulary is 50,257 tokens, every token
ID fits comfortably within `uint16` (max 65,535), which would cut the file to
**~900 MB** — a 4x reduction.

## Decision

Change the tokenized chunk storage dtype from `int64` to `uint16` in
`DataPipeline.py`. Cast back to `torch.long` in `__getitem__` so that
downstream consumers (`nn.Embedding`, loss functions) receive the dtype they
expect.

### Implementation details

**`DataPipeline.py`**:
- `np.array(ids, dtype=np.int64)` → `np.array(ids, dtype=np.uint16)` (line 119,
  per-story tokenization).
- `np.empty(total, dtype=np.int64)` → `np.empty(total, dtype=np.uint16)` (line
  126, concatenation buffer).
- `__getitem__` applies `.long()` to both `input_ids` and `target_ids` before
  returning. This is a zero-copy view on CPU and a negligible cast on GPU —
  the cost is invisible relative to a forward pass.

**Existing cache files**: Stale `int64` caches must be deleted so they rebuild
as `uint16`. There is no migration path — delete and rebuild is the intended
workflow.

## Consequences

### Positive
- `train_ctx256.pt` shrinks from ~3.5 GB to ~900 MB (4x), making it practical
  to store on the dashboard droplet and SCP to remote GPU instances.
- Eliminates the slowest step in remote job initialization (HF download +
  tokenization of 2.1M stories).
- Reduces memory footprint of `self.chunks` tensor at rest (though working
  memory is dominated by the model, not the dataset).

### Negative
- Requires deleting existing `int64` cache files on all machines (remote GPU,
  local dev). `torch.load` will succeed on old caches but `test_chunks_stored_as_uint16`
  will fail, surfacing the staleness clearly.
- `uint16` max is 65,535. If a future tokenizer has a vocabulary exceeding
  65,535 tokens, this will silently overflow. GPT-2 BPE (50,257) has ample
  headroom; a switch to a larger-vocab tokenizer would require bumping to
  `uint32` (~1.75 GB) or `int32`.

### Neutral
- The `.long()` cast in `__getitem__` means all downstream code (model,
  evaluator, generator) is unaffected — they still receive `torch.long`
  tensors. No changes required outside `DataPipeline.py`.

## Alternatives Considered

### int32 (4 bytes per token)
Halves the file from 3.5 GB to ~1.75 GB. Safer for future tokenizers with
vocab > 65K but wastes 2 bytes per token for GPT-2's 50K vocab. Rejected
because `uint16` gives a further 2x saving and GPT-2 is the only tokenizer
in use. Easy to revisit if the tokenizer changes.

### int16 (signed, 2 bytes per token)
Same size as `uint16` but max value is 32,767 — below GPT-2's vocab size of
50,257. Would silently corrupt token IDs above 32,767. Rejected.

### Compress with gzip/lz4 on disk, decompress on load
Could achieve similar or better compression ratios regardless of dtype. Rejected
because it adds decompression latency on every load, requires a dependency
(or custom wrapper), and the uint16 approach is simpler while achieving a
sufficient size reduction.

### Store on object storage (S3/DO Spaces) instead of the droplet
Decouples storage from the droplet. Not rejected outright — this is orthogonal
to the dtype decision and can be added later. The dtype change is worth doing
regardless because it reduces transfer time and storage cost everywhere.

## References

- PR: [#7 — Use uint16 for tokenized data cache to cut file size 4x](https://github.com/Michelleeby/tidal-language-model/pull/7)
- Code: `plugins/tidal/DataPipeline.py`
- Test: `plugins/tidal/tests/test_DataPipeline.py`
- ADR: [0001 — Single Modulation Gate](0001-single-modulation-gate.md)
