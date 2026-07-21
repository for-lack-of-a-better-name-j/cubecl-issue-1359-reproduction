use burn::backend::rocm::RocmDevice;
use burn::backend::{Autodiff, Rocm};
use burn::prelude::*;

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
fn main() {
    type RocmBackend = Autodiff<Rocm>;
    let device = RocmDevice::default();
    trigger_overflow_burn_multiple_tensors::<RocmBackend>(&device);
}
