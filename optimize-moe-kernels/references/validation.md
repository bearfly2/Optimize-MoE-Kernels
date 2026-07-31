# MoE Kernel Validation

## Contents

1. Correctness
2. Alignment and safety
3. Event audit
4. Generated-source audit
5. Timeline audit
6. Benchmark discipline
7. Reporting template

## 1. Correctness

- Compare against the authoritative framework or AscendC operator, including the official gradient operator when available.
- Cover every dispatch path and optional-input combination.
- Include K values used in production plus unsupported-K fallback cases.
- Include invalid routes (`-1`), tail tokens, non-divisible core partitions, and duplicate positions if permitted.
- Compare every output independently: activations, route weights, scale gradients, indices, and metadata.
- Accumulate sums and dot products in FP32 before output casting.
- Report error count, ratio, max absolute error, and the first mismatching indices during debugging.

## 2. Alignment and Safety

For every DMA, record source offset, destination offset, element size, and transfer length.

- Reject short unaligned metadata copies.
- Verify the final owner and final block cannot read beyond GM.
- Do not rely on `pad_value` to make an out-of-bounds transfer safe when safe-memory lowering is disabled.
- Keep intrinsic arguments as full/stable Buffers when lowering requires `access_ptr()`.
- Inspect tail-specific branches separately from full blocks.

## 3. Event Audit

For each event and stage, write the lifecycle:

```text
initial producer -> consumer -> work -> release producer -> next consumer
```

Check:

- Every initialized event is consumed.
- Every produced event has exactly one consumer before reuse.
- Empty owners and final partial blocks follow a complete event path.
- A buffer is not overwritten until its last consumer pipeline has finished.
- Automatic synchronization settings preserve required scalar/Vector dependencies.

Treat a hang as an event/dependency bug until proven otherwise. Restart the process after an AICore exception or timeout before retesting.

## 4. Generated-Source Audit

Use kernel-source printing only when needed. Confirm:

- The expected dispatch path compiled.
- Compile-time branches and K loops were removed/unrolled.
- DMA lengths and address arithmetic are aligned.
- Double-buffer stages have distinct UB addresses.
- Prefetch instructions occur before the blocking wait they are meant to hide.
- No unexpected full-pipeline barrier was inserted.
- No optional input is read in a path where it is absent.

Source order is evidence, not proof of runtime overlap.

## 5. Timeline Audit

Inspect one AIV at sufficient zoom and identify block boundaries.

- Distinguish metadata DMA, row DMA, Vector work, scalar waits, and output DMA.
- Confirm the next block's operation belongs to the intended stage, not merely a small metadata prefetch.
- Measure overlap duration, exposed MTE2/MTE3 tails, wait duration, and command gaps.
- Verify the timeline shape across K, H, dtype, and optional-input paths.
- If events exist but overlap does not, identify the command-issue dependency rather than adding more events.

## 6. Benchmark Discipline

- Benchmark TileLang and AscendC independently; avoid back-to-back measurements that alter cache, stream, or frequency state.
- Warm up compilation and execution separately.
- Use repeated samples and report median plus variability.
- Keep input generation and ACLNN setup outside the measured interval.
- Compare the same tensor layout, dtype, mapping distribution, and synchronization policy.
- Track results by scenario, not only a global average.
- Estimate effective bytes/time, but compare random gather/scatter against a realistic indirect-access roof rather than peak contiguous bandwidth.
- Reject an optimization that improves a timeline screenshot but does not improve stable latency.

## 7. Reporting Template

```text
Scenario:
Dispatch path:
Correctness reference:
Dominant engine before/after:
Bytes and operations per token:
Tiling and UB usage:
Generated-source findings:
Timeline findings:
Latency before/after:
Regressions:
Decision: keep / revise / revert
```

