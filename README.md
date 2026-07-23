# cubecl-issue-1359-reproduction
Minimal reproduction harness and analysis for [CubeCL Issue #1359-Hitting addition with overflow in FlushingPolicyState](https://github.com/tracel-ai/cubecl/issues/1359).

## Summary
In `cubecl-hip` version 0.10.0, multiple tensor allocations that are **less than 4.29 GiB individually** but **more than 4.29GiB total** can cause this error, originally pointed out by `jeandudey`  in [CubeCL issue #1359](https://github.com/tracel-ai/cubecl/issues/1359):
```
thread 'DSD-0-0' (13245) panicked at /home/j/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/cubecl-runtime-0.10.0/src/memory_management/drop_queue/policy.rs:36:9:
attempt to add with overflow
```
I found out the specific conditions under which this bug could be reproduced by learning how CubeCL allocates memory.

### Environment
- `cubecl-runtime v0.10.0`
- `cubecl-hip v0.10.0`, also applies to `cubecl-cuda v0.10.0`, but using AMD ROCm here.

## Failure Mechanism

<!--The part of the code that panicked was in the `cubecl-runtime` crate, in the `register` function of `FlushingPolicyState`:-->
The panic originates inside the host-side resource tracking layer of the `cubecl-runtime` crate, specifically within the `register` function of `FlushingPolicyState`:
```rust
/// Tracks staged allocations and evaluates them against a [`FlushingPolicy`].
#[derive(Default, Debug)]
pub(crate) struct FlushingPolicyState {
    bytes_count: u32,
    bytes_size: u32,
}

impl FlushingPolicyState {
    /// Record a newly staged [`Bytes`] allocation.
    pub(crate) fn register(&mut self, bytes: &Bytes) {
        self.bytes_count += 1;
        self.bytes_size += bytes.len() as u32;
    }
}
```
Since the panic happened on both WSL on ROCm as well as CUDA per the issue, it was clear that the issue was vendor-agnostic. To reproduce the bug, I used `lldb` to inspect state at the time of the crash:
![LLDB showing bytes_size overflow](reproduced_the_bug_with_bytes_size.png)
### Code to Reproduce
Since the bug triggers when multiple tensors, individually less than 4.29GiB, but collectively more than 4.29GiB were written to the GPU between kernel launches, I wrote a minimal reproduction function:
```rust
fn trigger_overflow_burn_multiple_tensors<B: Backend>(device: &B::Device) {
    let mut tensors = Vec::new();
    let shape = [625_000_000]; // 625,000,000 f32s * 4 bytes = 2.5gb,
    println!("Big data dump into allocation queue");
    for i in 0..3 {
        println!("creating tensor {i}");
        let data = TensorData::ones::<f32, _>(shape);
        let tensor = Tensor::<B, 1>::from_data(data, device);
        tensors.push(tensor);
    }
    println!("chain some ops, force load to GPU and make it crash");
    let mut computation = tensors[0].clone();

    println!("The overflow will happen about here.");
    for tensor in tensors.iter().skip(1) {
        computation = computation * tensor.clone();
    }
    let raw_data = computation.into_data();
    println!("raw_data: {:?}", raw_data);
}
```
In this function, three tensors of 2.5GiB are allocated. CubeCL uses lazy evaluation, so it does not send tensors to the GPU until they are used in the multiplication, so the panic does not happen until the multiplication occurs. Since CubeCL does not check to flush on each tensor allocation, and only checks in GPU kernel launches, the loop causes `FlushingPolicyState.bytes_size` to overflow with >5GiB of allocations before the next kernel launch can flush it.

## Root Cause Analysis
If you're reading this and you're still interested, this is the part where I'll
talk a little bit more about my thought process that I used to find the root 
cause of the bug. I knew that a simple type widening of a value may have been a
band-aid fix because developers often build little sanity checks into their 
code. NASA for example uses lots of assert statements: "If this assert breaks, 
the assumptions I made while writing this code are broken and we need to do 
something about it." So I started thinking that the fact that the `u32` type
in `FlushingPolicyState.bytes_size` was intentional. 

However, I needed to be sure somehow. I had never done any deep debugging like 
this in Rust before, so I started off flailing  with `println!` macros for three 
weeks. I tried everything I could think of--but since CubeCL is an async runtime
with many moving parts, I eventually realized that `println!` just wasn't 
going to cut the mustard. I realized it would behoove me to see if the Rust 
ecosystem had a good debugger. Turns out it did; so I set up `lldb` in my
AstroNvim installation with the defaults from the 
`astronvim-community` repository. 

Once I set up `lldb` figuring out how the system worked was much easier. For a 
while there, it felt like Sisyphus pushing on that boulder in Tartarus--I had 
chosen a bit of a challenging task to start with in the low-level space, but
I just knew there had to be some mechanistic reason for the `u32` overflow.

