use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rust_neural_networks::layers::{Conv2DLayer, Layer};
use rust_neural_networks::utils::rng::SimpleRng;

fn bench_conv2d_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d_cpu");

    // CNN-like configuration (common in ResNet-style models).
    // Note: Conv2DLayer expects NHWC input layout.
    let (n, h, w, cin, cout, k, stride, pad) = (
        32usize, 56usize, 56usize, 64usize, 128usize, 3usize, 1usize, 1isize,
    );

    let mut rng = SimpleRng::new(123);
    let mut layer = Conv2DLayer::new(cin, cout, k, pad, stride, h, w, &mut rng);

    let input_size = layer.input_size();
    let output_size = layer.output_size();

    let mut input = vec![0.0f32; n * input_size];
    let mut grad_output = vec![0.0f32; n * output_size];

    for x in &mut input {
        *x = rng.next_f32();
    }
    for x in &mut grad_output {
        *x = rng.next_f32();
    }

    let mut output = vec![0.0f32; n * output_size];
    let mut grad_input = vec![0.0f32; n * input_size];

    // Forward
    group.throughput(Throughput::Elements((n * h * w * cin) as u64));
    group.bench_with_input(
        BenchmarkId::new(
            "forward",
            format!("N{} C{}->{} {}x{} k{}", n, cin, cout, h, w, k),
        ),
        &n,
        |b, &batch_size| {
            b.iter(|| {
                layer.forward(black_box(&input), black_box(&mut output), batch_size);
                black_box(&output);
            })
        },
    );

    // Backward
    group.bench_with_input(
        BenchmarkId::new(
            "backward",
            format!("N{} C{}->{} {}x{} k{}", n, cin, cout, h, w, k),
        ),
        &n,
        |b, &batch_size| {
            b.iter(|| {
                layer.backward(
                    black_box(&input),
                    black_box(&grad_output),
                    black_box(&mut grad_input),
                    batch_size,
                );
                black_box(&grad_input);
            })
        },
    );

    // Training step (forward + backward + update)
    group.bench_with_input(
        BenchmarkId::new(
            "train_step",
            format!("N{} C{}->{} {}x{} k{}", n, cin, cout, h, w, k),
        ),
        &n,
        |b, &batch_size| {
            b.iter(|| {
                layer.forward(black_box(&input), black_box(&mut output), batch_size);
                layer.backward(
                    black_box(&input),
                    black_box(&grad_output),
                    black_box(&mut grad_input),
                    batch_size,
                );
                layer.update_parameters(black_box(0.01));
                black_box((&output, &grad_input));
            })
        },
    );

    group.finish();
}

criterion_group!(benches, bench_conv2d_cpu);
criterion_main!(benches);
