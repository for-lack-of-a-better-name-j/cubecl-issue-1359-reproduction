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
fn allocate_humongous_tensor<B: Backend>(device: &B::Device) {
    let shape = [1_250_000_000]; // 1,250,000,000 f32s * 4 bytes = 5.0gb,
    println!("Creating humongous tensor!");
    let humongous_data = TensorData::ones::<f32, _>(shape);
    let humongous_tensor = Tensor::<B, 1>::from_data(humongous_data, device);
    println!("Creating itty bitty tensor!");
    let itty_bitty_shape = [1]; //
    let mut itty_bitty_tensors = Vec::new();
    for i in 0..3 {
        println!("creating itty_bitty_tensor {i}");
        let itty_bitty_data = TensorData::ones::<f32, _>([1]);
        let itty_bitty_tensor = Tensor::<B, 1>::from_data(itty_bitty_data, device);
        itty_bitty_tensors.push(itty_bitty_tensor.clone());
    }

    println!("chain some ops so that they are sent to GPU and show large allocation behavior!");
    let mut computation = itty_bitty_tensors[0].clone();
    for itty_bitty_tensor in itty_bitty_tensors.iter().skip(1) {
        computation = computation * humongous_tensor.clone();
    }
    let raw_data = computation.into_data();
    println!("raw_data: {:?}", raw_data);
}
fn main() {
    type RocmBackend = Autodiff<Rocm>;
    let device = RocmDevice::default();
    allocate_humongous_tensor::<RocmBackend>(&device);
    //trigger_overflow_burn_multiple_tensors::<RocmBackend>(&device);
}
