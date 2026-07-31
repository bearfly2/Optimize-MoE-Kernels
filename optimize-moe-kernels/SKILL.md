---
name: optimize-moe-kernels
description: Analyze, implement, and validate MoE kernel optimizations, especially for Ascend, TileLang-Ascend, and CANN. Use for routing/top-k, group count, dispatch/expand, combine/reduce, forward/backward gradients, indirect gather/scatter, small-K specialization, alignment, UB tiling, buffering, pipeline analysis, precision comparison, generated-source inspection, and performance benchmarking.
---

# Optimize MoE Kernels

Treat MoE optimization as a data-layout and dependency problem before treating it as an instruction-scheduling problem.

## Workflow

1. Establish the exact contract.
   - Record tensor shapes, dtypes, optional inputs, valid `K` values, invalid-route sentinel, duplicate-position semantics, and required outputs.
   - Identify whether the framework exposes token-to-position, position-to-token, expert offsets, sorted routes, or only one mapping direction.

2. Classify the operator by dominant dataflow.
   - Routing/top-k: Vector/control and selection dominated.
   - Count/prefix/metadata: scalar, histogram, reduction, or atomic dominated.
   - Dispatch/expand: read one token row and scatter to K expanded rows; usually MTE3 dominated.
   - Combine/reduce: gather K expanded rows and reduce to one token row; usually MTE2 dominated.
   - Backward: split unweighted replication from weighted `dx + dweight` fusion.

3. Build a byte and operation model before editing.
   - Count GM bytes for metadata, row reads, row writes, and optional tensors.
   - Count Vector operations and reductions separately.
   - Decide whether the expected bound is MTE2, Vector, MTE3, scalar issue, or load imbalance.
   - When generated code or profiling shows scalar issue pressure, serial gather/index loops, scalar reduce chains, or manual broadcast expansion, invoke `$tilelang-scalar-optimization` before choosing a transformation.

4. Choose ownership and layout first.
   - Prefer token ownership for token reductions and reuse of one `dout` row.
   - Prefer expanded-position ownership for contiguous `dx` writes when a reverse mapping exists.
   - Prefer route-count-based work assignment for expert-skewed workloads.
   - Propose an upstream metadata/layout change when indirect addresses prevent the desired DMA pattern.

5. Select specialization axes conservatively.
   - Specialize common small K values at compile time and unroll K.
   - Separate paths by behavior: unweighted/unscaled, weighted/unscaled, scaled, mixed dtype, and fallback.
   - Tune rows per block and route stages from UB usage and H; do not assume four buffers always win.
   - Keep a generic correctness fallback for unsupported shapes.

6. Implement aligned block operations.
   - Move metadata in aligned multi-token blocks, not per route.
   - Keep intrinsic operands as stable one-dimensional Buffers when lowering requires `access_ptr()`.
   - Handle tails explicitly; never assume padding protects an out-of-bounds DMA when safe-memory lowering is disabled.

7. Add overlap only where dependencies permit it.
   - Use distinct physical UB stages and balanced event cycles.
   - Place the next DMA before the first control wait that would prevent its issue.
   - Do not claim overlap from flags or source structure alone; inspect generated source and device timelines.
   - Treat dynamic gather/scatter addresses as a possible serialization boundary.

8. Validate in four layers.
   - Precision against the authoritative implementation.
   - Generated source for actual DMA sizes, buffer addresses, waits, and branch removal.
   - Device timeline for real MTE2/Vector/MTE3 overlap and stalls.
   - Isolated performance with warmup, repeated samples, and per-scenario regression checks.

## Scalar-Heavy TileLang Escalation

When a TileLang-Ascend MoE kernel is already correct but its hot path is scalar-heavy, use `$tilelang-scalar-optimization` as a focused sub-workflow:

1. Lock the exact operator repository, revision, kernel files, builders, and symbols from the contract. Do not substitute an adjacent MoE operator because it has similar scalar loops; if the target source is absent at that revision, stop and report it.
2. Preserve and identify the last committed precision-passing baseline.
3. Inspect successful target-file history before editing; do not treat dirty or failed experiments as evidence.
4. Read that skill's `scalar-patterns.md` and select the smallest applicable scalar-to-tile transformation.
5. Preserve generic fallbacks for shape- or layout-specific fast paths.
6. Re-run precision first, then inspect generated Ascend C and profile to confirm scalar work became tile/vector work.
7. Record the exact target symbols, baseline revision, candidate revision, selected pattern, precision result, generated-source evidence, and scenario-level timing in the optimization report.

Do not invoke the scalar skill merely because scalar expressions exist. Keep compile-time layout choices, loop bounds, offsets, and guards scalar unless device evidence identifies them as the bottleneck.

9. Report results by scenario.
   - Name kernels by their dispatch conditions, not by vague terms such as `specialized` or `compact`.
   - State which cases improved, regressed, or were unchanged.
   - Revert unsuccessful experiments instead of accumulating inactive complexity.

## Decision Rules

- Prefer changing route ownership or metadata layout over adding more flags to an inherently serialized indirect-access loop.
- Prefer whole-row transfer when the row fits UB and is naturally aligned; tile H only when UB pressure or instruction limits require it.
- Reuse loaded rows across K routes and fuse outputs that share the same inputs.
- Keep accumulation and dot products in FP32 unless the contract explicitly permits lower precision.
- Preserve compile-time unrolling when using macros to reduce source duplication.
- Do not infer bandwidth efficiency from nominal bytes alone; random GM accesses have a lower attainable roof than contiguous DMA.

## References

- Read [references/patterns.md](references/patterns.md) for operator-specific optimization patterns and known limitations.
- Read [references/validation.md](references/validation.md) before changing events, alignment, precision logic, or benchmark code.
