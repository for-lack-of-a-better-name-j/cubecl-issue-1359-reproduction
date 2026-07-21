# cubecl-issue-1359-reproduction
Minimal reproduction harness and analysis for [CubeCL Issue #1359-Hitting addition with overflow in FlushingPolicyState](https://github.com/tracel-ai/cubecl/issues/1359).

## Summary
In `cubecl-hip` version 0.10.0, multiple tensor allocations that are **less than 4.2 GiB individually** but **more than 4.2GiB total** can cause this error, originally pointed out by `jeandudey`  in [CubeCL issue #1359](https://github.com/tracel-ai/cubecl/issues/1359):
```
thread 'DSD-0-0' (13245) panicked at /home/j/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-runtime-0.10.0/src/memory_management/drop_queue/policy.rs:36:9:
attempt to add with overflow
```
I found out the specific conditions under which this bug could be reproduced by learning how CubeCL allocates memory.

### Environment
- `cubecl-runtime v0.10.0`
- `cubecl-hip v0.10.0`, also applies to `cubecl-cuda v0.10.0`, but using AMD ROCm here.

