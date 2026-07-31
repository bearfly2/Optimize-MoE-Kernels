# MoE Optimization Patterns

## Contents

1. Shared characteristics
2. Routing and top-k
3. Count and prefix metadata
4. Dispatch and expand
5. Combine and reduce
6. Backward
7. Tiling and alignment
8. Pipeline limits
9. Cross-operator opportunities

## 1. Shared Characteristics

MoE kernels commonly combine a small static K, a long hidden row H, indirect route mappings, optional weights/scales, and skewed expert load. Arithmetic intensity is usually low outside routing selection and weighted backward dot products.

Optimize the full route lifecycle:

```text
scores -> top-k -> rank/expert mapping -> counts/offsets
       -> dispatch/expand -> expert compute -> combine/reduce
       -> backward scatter/gather and route gradients
```

Producing better metadata once can remove random access and repeated conversion from several downstream kernels.

## 2. Routing and Top-K

Lessons from `top2_sum_gate`:

- Use the hardware top-k intrinsic instead of scalar insertion/sort loops when its supported shapes fit.
- Align top-k source storage to the intrinsic requirement, commonly 32 elements; fill padding with negative infinity while passing the real element count.
- Fuse bias, scoring transform, group filtering, top-k, normalization, and rank mapping when intermediates are not reused elsewhere.
- Isolate grouped-routing slow paths so improvements do not regress the common skip-group path.
- For grouped top-k, reduce candidate complexity from repeated `experts * selected_groups` scans to one expert scan plus compact group/candidate selection.
- Load static maps and bias once per AIV or cache them across calls when the runtime permits it.
- Batch several token rows in UB when selection intrinsics and UB capacity allow it.

## 3. Count and Prefix Metadata

Avoid algorithms where every output token rescans all routes. Prefer:

- Per-core local histograms followed by reduction.
- Atomics only when contention is lower than the cost of a second reduction stage.
- Aligned group/expert counters in UB.
- Fusing count, prefix offset, and route-list construction when intermediate counts have no independent consumer.
- Work assignment based on route count rather than expert count when routing is skewed.

## 4. Dispatch and Expand

The usual forward operation reads one token row and writes it to K expanded positions.

- Load the token row once and reuse it for all valid routes.
- Load K positions as one aligned metadata block.
- Expect MTE3 to dominate because output bytes are roughly K times input bytes.
- Random destinations prevent one contiguous DMA. If upstream can emit routes ordered by destination, use expanded-position ownership and contiguous writes.
- Pipeline next-token MTE2 with current-token MTE3 only after confirming dynamic destination calculation does not block command issue.

The backward of expand is the inverse: gather K expanded gradient rows and sum one token row. Apply combine/reduce patterns.

## 5. Combine and Reduce

The usual forward operation gathers K expanded rows and reduces them to one token row.

- Use token ownership to avoid output atomics.
- Unroll common K values and rotate two or four x buffers.
- Keep the accumulator in FP32; cast only at boundaries.
- Load weights and mappings in aligned multi-token blocks.
- Reuse one output accumulator across routes and perform one whole-row writeback.
- For weighted paths, fuse scale construction with row multiply/add instead of materializing weight vectors repeatedly.
- Expect random MTE2 gather to dominate; increasing Vector parallelism cannot hide an address-issue bottleneck by itself.

## 6. Backward

Split by mathematical behavior:

### Unweighted and unscaled

```text
dx[pos_k, :] = dout[token, :]
```

Read `dout` once and replicate it K times. This is a random MTE3 scatter path. The best structural optimization is often to provide `pos_to_token` and let each worker own contiguous expanded positions.

### Weighted

```text
dx[pos_k, :] = dout[token, :] * weight[token, k]
dweight[token, k] = dot(dout[token, :], x[pos_k, :])
```

- Fuse both outputs to reuse `dout` and `x`.
- Batch two tokens for K=2 to expose four routes.
- Use two/four x and dx stages according to UB capacity.
- Overlap the next x gather with current Vector work and dx writeback where the generated code permits it.

### Scale gradients

- Accumulate local scalar reductions per AIV.
- Reduce local partials in a second stage or with controlled atomics.
- Avoid assigning a full token/route scan to one task.

Assume direct writes only when every expanded position has one owner. Use atomics or deduplicate when mappings may contain repeated valid positions.

## 7. Tiling and Alignment

Useful starting points, not universal constants:

```text
H <= 2048: try 4 token rows per block
H >  2048: try 2 token rows per block
H <= 3072: try 4 route stages
H >  3072: try 2 route stages
```

Calculate UB usage explicitly:

```text
metadata
+ token_rows * dout_row
+ route_stages * x_row
+ route_stages * dx_row
+ cast/reduction temporaries
```

Alignment rules:

- Build DMA blocks in multiples of the hardware transfer unit, commonly 32 bytes.
- Eight FP32/int32 values or sixteen FP16/BF16 values occupy 32 bytes.
- Do not issue per-route short metadata DMA for small K; pack multiple tokens.
- Separate source-address alignment, destination-address alignment, and transfer-length alignment.
- Use one-dimensional staging buffers when a TileLang intrinsic requires a Buffer with `access_ptr()`.
- Match `real_shape` rank to the intrinsic operand rank.

## 8. Pipeline Limits

A valid conceptual pipeline is:

```text
MTE2(N+1) | Vector(N) | MTE3(N-1)
```

It requires independent buffers, early command issue, and correct event ownership. Common failure modes:

- The next DMA is written after a wait/barrier that already serialized the control path.
- Memory planning aliases two apparent stages.
- Automatic synchronization closes a manually created overlap window.
- Disabling automatic synchronization removes a scalar/Vector dependency and hangs the kernel.
- Dynamic route addresses delay MTE3 issue until scalar control completes.
- Events are initialized but not consumed on tail/empty branches.

Do not expect MTE3 operations on one AIV to execute concurrently with each other. The goal is to hide other engines under a long MTE3 sequence.

## 9. Cross-Operator Opportunities

- Generate both `token_to_pos` and `pos_to_token` during routing.
- Reuse sorted routes, counts, expert offsets, and rank maps across dispatch, combine, and backward.
- Preallocate outputs/workspaces and avoid wrapper-side ACLNN initialization in measured paths.
- Fuse cheap pointwise transforms into the nearest bandwidth-bound route kernel.
- Preserve separate fast paths for common no-weight/no-scale cases.
- Keep uncommon combinations on a correct generic fallback until profiling justifies specialization.

