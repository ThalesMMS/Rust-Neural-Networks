use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use rust_neural_networks::data::cifar10::read_cifar10_batches;

const TRAINING_BATCHES: [&str; 5] = [
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/cifar-10-batches-bin/data_batch_1.bin"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/cifar-10-batches-bin/data_batch_2.bin"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/cifar-10-batches-bin/data_batch_3.bin"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/cifar-10-batches-bin/data_batch_4.bin"
    ),
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/data/cifar-10-batches-bin/data_batch_5.bin"
    ),
];

fn cifar10_loader_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cifar10_loader");
    group.throughput(Throughput::Elements(50_000));

    group.bench_function("read_training_batches", |b| {
        b.iter(|| {
            // Requires CIFAR-10 binary files under data/cifar-10-batches-bin/.
            let data = read_cifar10_batches(black_box(&TRAINING_BATCHES))
                .expect("failed to read CIFAR-10 training batches");
            black_box(data);
        });
    });

    group.finish();
}

criterion_group!(benches, cifar10_loader_benchmark);
criterion_main!(benches);
