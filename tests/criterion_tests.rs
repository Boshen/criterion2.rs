use std::{
    cell::{Cell, RefCell},
    cmp::max,
    fs::File,
    path::{Path, PathBuf},
    rc::Rc,
    time::{Duration, SystemTime},
};

use criterion::{BatchSize, BenchmarkFilter, BenchmarkId, Criterion, profiler::Profiler};
use serde_json::value::Value;
use tempfile::{TempDir, tempdir};
use walkdir::WalkDir;

/*
 * Please note that these tests are not complete examples of how to use
 * Criterion.rs. See the benches folder for actual examples.
 */
fn temp_dir() -> TempDir {
    tempdir().unwrap()
}

// Configure a Criterion struct to perform really fast benchmarks. This is not
// recommended for real benchmarking, only for testing.
fn short_benchmark(dir: &TempDir) -> Criterion {
    Criterion::default()
        .output_directory(dir.path())
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_millis(500))
        .nresamples(2000)
}

#[derive(Clone)]
struct Counter {
    counter: Rc<RefCell<usize>>,
}
impl Counter {
    fn count(&self) {
        *(*self.counter).borrow_mut() += 1;
    }

    fn read(&self) -> usize {
        *(*self.counter).borrow()
    }
}
impl Default for Counter {
    fn default() -> Counter {
        Counter { counter: Rc::new(RefCell::new(0)) }
    }
}

fn verify_file(dir: &Path, path: &str) -> PathBuf {
    let full_path = dir.join(path);
    assert!(full_path.is_file(), "File {full_path:?} does not exist or is not a file");
    let metadata = full_path.metadata().unwrap();
    assert!(metadata.len() > 0);
    full_path
}

fn verify_json(dir: &Path, path: &str) {
    let full_path = verify_file(dir, path);
    let f = File::open(full_path).unwrap();
    serde_json::from_reader::<File, Value>(f).unwrap();
}

fn verify_stats(dir: &Path, baseline: &str) {
    verify_json(dir, &format!("{baseline}/estimates.json"));
    verify_json(dir, &format!("{baseline}/sample.json"));
    verify_json(dir, &format!("{baseline}/tukey.json"));
    verify_json(dir, &format!("{baseline}/benchmark.json"));
}

fn verify_not_exists(dir: &Path, path: &str) {
    assert!(!dir.join(path).exists());
}

fn latest_modified(dir: &Path) -> SystemTime {
    let mut newest_update: Option<SystemTime> = None;
    for entry in WalkDir::new(dir) {
        let entry = entry.unwrap();
        let modified = entry.metadata().unwrap().modified().unwrap();
        newest_update = match newest_update {
            Some(latest) => Some(max(latest, modified)),
            None => Some(modified),
        };
    }

    newest_update.expect("failed to find a single time in directory")
}

#[test]
fn test_creates_directory() {
    let dir = temp_dir();
    short_benchmark(&dir).bench_function("test_creates_directory", |b| b.iter(|| 10));
    assert!(dir.path().join("test_creates_directory").is_dir());
}

#[test]
fn test_without_plots() {
    let dir = temp_dir();
    short_benchmark(&dir).bench_function("test_without_plots", |b| b.iter(|| 10));

    for entry in WalkDir::new(dir.path().join("test_without_plots")) {
        let entry = entry.ok();
        let is_svg = entry
            .as_ref()
            .and_then(|entry| entry.path().extension())
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "svg")
            .unwrap_or(false);
        assert!(
            !is_svg,
            "Found SVG file ({:?}) in output directory with plots disabled",
            entry.unwrap().file_name()
        );
    }
}

#[test]
fn test_save_baseline() {
    let dir = temp_dir();
    println!("tmp directory is {:?}", dir.path());
    short_benchmark(&dir)
        .save_baseline("some-baseline".to_owned())
        .bench_function("test_save_baseline", |b| b.iter(|| 10));

    let dir = dir.path().join("test_save_baseline");
    verify_stats(&dir, "some-baseline");

    verify_not_exists(&dir, "base");
}

#[test]
fn test_retain_baseline() {
    // Initial benchmark to populate
    let dir = temp_dir();
    short_benchmark(&dir)
        .save_baseline("some-baseline".to_owned())
        .bench_function("test_retain_baseline", |b| b.iter(|| 10));

    let pre_modified = latest_modified(&dir.path().join("test_retain_baseline/some-baseline"));

    short_benchmark(&dir)
        .retain_baseline("some-baseline".to_owned(), true)
        .bench_function("test_retain_baseline", |b| b.iter(|| 10));

    let post_modified = latest_modified(&dir.path().join("test_retain_baseline/some-baseline"));

    assert_eq!(pre_modified, post_modified, "baseline modified by retain");
}

#[test]
#[should_panic(expected = "Baseline 'some-baseline' must exist before comparison is allowed")]
fn test_compare_baseline_strict_panics_when_missing_baseline() {
    let dir = temp_dir();
    short_benchmark(&dir)
        .retain_baseline("some-baseline".to_owned(), true)
        .bench_function("test_compare_baseline", |b| b.iter(|| 10));
}

#[test]
fn test_compare_baseline_lenient_when_missing_baseline() {
    let dir = temp_dir();
    short_benchmark(&dir)
        .retain_baseline("some-baseline".to_owned(), false)
        .bench_function("test_compare_baseline", |b| b.iter(|| 10));
}

#[test]
fn test_sample_size() {
    let dir = temp_dir();
    let counter = Counter::default();

    let clone = counter.clone();
    short_benchmark(&dir).sample_size(50).bench_function("test_sample_size", move |b| {
        clone.count();
        b.iter(|| 10)
    });

    // This function will be called more than sample_size times because of the
    // warmup.
    assert!(counter.read() > 50);
}

#[test]
fn test_warmup_time() {
    let dir = temp_dir();
    let counter1 = Counter::default();

    let clone = counter1.clone();
    short_benchmark(&dir).warm_up_time(Duration::from_millis(100)).bench_function(
        "test_warmup_time_1",
        move |b| {
            clone.count();
            b.iter(|| 10)
        },
    );

    let counter2 = Counter::default();
    let clone = counter2.clone();
    short_benchmark(&dir).warm_up_time(Duration::from_millis(2000)).bench_function(
        "test_warmup_time_2",
        move |b| {
            clone.count();
            b.iter(|| 10)
        },
    );

    assert!(counter1.read() < counter2.read());
}

#[test]
fn test_measurement_time() {
    let dir = temp_dir();
    let counter1 = Counter::default();

    let clone = counter1.clone();
    short_benchmark(&dir)
        .measurement_time(Duration::from_millis(100))
        .bench_function("test_meas_time_1", move |b| b.iter(|| clone.count()));

    let counter2 = Counter::default();
    let clone = counter2.clone();
    short_benchmark(&dir)
        .measurement_time(Duration::from_millis(2000))
        .bench_function("test_meas_time_2", move |b| b.iter(|| clone.count()));

    assert!(counter1.read() < counter2.read());
}

#[test]
fn test_bench_function() {
    let dir = temp_dir();
    short_benchmark(&dir).bench_function("test_bench_function", move |b| b.iter(|| 10));
}

#[test]
fn test_filtering() {
    let dir = temp_dir();
    let counter = Counter::default();
    let clone = counter.clone();

    short_benchmark(&dir)
        .with_benchmark_filter(BenchmarkFilter::Exact("Foo".into()))
        .bench_function("test_filtering", move |b| b.iter(|| clone.count()));

    assert_eq!(counter.read(), 0);
    assert!(!dir.path().join("test_filtering").is_dir());
}

#[test]
fn test_timing_loops() {
    let dir = temp_dir();
    let mut c = short_benchmark(&dir);
    let mut group = c.benchmark_group("test_timing_loops");
    group.bench_function("iter_with_setup", |b| b.iter_with_setup(|| vec![10], |v| v[0]));
    group.bench_function("iter_with_large_setup", |b| {
        b.iter_batched(|| vec![10], |v| v[0], BatchSize::NumBatches(1))
    });
    group.bench_function("iter_with_large_drop", |b| b.iter_with_large_drop(|| vec![10; 100]));
    group.bench_function("iter_batched_small", |b| {
        b.iter_batched(|| vec![10], |v| v[0], BatchSize::SmallInput)
    });
    group.bench_function("iter_batched_large", |b| {
        b.iter_batched(|| vec![10], |v| v[0], BatchSize::LargeInput)
    });
    group.bench_function("iter_batched_per_iteration", |b| {
        b.iter_batched(|| vec![10], |v| v[0], BatchSize::PerIteration)
    });
    group.bench_function("iter_batched_one_batch", |b| {
        b.iter_batched(|| vec![10], |v| v[0], BatchSize::NumBatches(1))
    });
    group.bench_function("iter_batched_10_iterations", |b| {
        b.iter_batched(|| vec![10], |v| v[0], BatchSize::NumIterations(10))
    });
    group.bench_function("iter_batched_ref_small", |b| {
        b.iter_batched_ref(|| vec![10], |v| v[0], BatchSize::SmallInput)
    });
    group.bench_function("iter_batched_ref_large", |b| {
        b.iter_batched_ref(|| vec![10], |v| v[0], BatchSize::LargeInput)
    });
    group.bench_function("iter_batched_ref_per_iteration", |b| {
        b.iter_batched_ref(|| vec![10], |v| v[0], BatchSize::PerIteration)
    });
    group.bench_function("iter_batched_ref_one_batch", |b| {
        b.iter_batched_ref(|| vec![10], |v| v[0], BatchSize::NumBatches(1))
    });
    group.bench_function("iter_batched_ref_10_iterations", |b| {
        b.iter_batched_ref(|| vec![10], |v| v[0], BatchSize::NumIterations(10))
    });
    let mut global_data: Vec<u8> = vec![];
    group.bench_function("iter_with_setup_wrapper", |b| {
        b.iter_with_setup_wrapper(|runner| {
            global_data.clear();
            global_data.extend(&[1, 2, 3, 4, 5]);
            let len = runner.run(|| {
                global_data.push(6);
                global_data.len()
            });
            assert_eq!(len, 6);
            assert_eq!(global_data.len(), 6);
        })
    });
}

#[test]
#[should_panic(expected = "Benchmark function must call Bencher::iter or related method.")]
fn test_bench_with_no_iteration_panics() {
    let dir = temp_dir();
    short_benchmark(&dir).bench_function("no_iter", |_b| {});
}

#[test]
#[should_panic(expected = "setup function must call `WrapperRunner::run`")]
fn test_setup_wrapper_with_no_runner_call_panics() {
    let dir = temp_dir();
    short_benchmark(&dir).bench_function("no_run_call", |b| {
        b.iter_with_setup_wrapper(|_runner| {});
    });
}

#[test]
#[should_panic(expected = "setup function must call `WrapperRunner::run` only once")]
fn test_setup_wrapper_with_multiple_runner_calls_panics() {
    let dir = temp_dir();
    short_benchmark(&dir).bench_function("no_run_call", |b| {
        b.iter_with_setup_wrapper(|runner| {
            runner.run(|| {});
            runner.run(|| {});
        });
    });
}

#[test]
fn test_benchmark_group_with_input() {
    let dir = temp_dir();
    let mut c = short_benchmark(&dir);
    let mut group = c.benchmark_group("Test Group");
    for x in 0..2 {
        group.bench_with_input(BenchmarkId::new("Test 1", x), &x, |b, i| b.iter(|| i));
        group.bench_with_input(BenchmarkId::new("Test 2", x), &x, |b, i| b.iter(|| i));
    }
    group.finish();
}

#[test]
fn test_benchmark_group_without_input() {
    let dir = temp_dir();
    let mut c = short_benchmark(&dir);
    let mut group = c.benchmark_group("Test Group 2");
    group.bench_function("Test 1", |b| b.iter(|| 30));
    group.bench_function("Test 2", |b| b.iter(|| 20));
    group.finish();
}

#[test]
fn test_criterion_doesnt_panic_if_measured_time_is_zero() {
    let dir = temp_dir();
    let mut c = short_benchmark(&dir);
    c.bench_function("zero_time", |bencher| bencher.iter_custom(|_iters| Duration::new(0, 0)));
}

struct TestProfiler {
    started: Rc<Cell<u32>>,
    stopped: Rc<Cell<u32>>,
}
impl Profiler for TestProfiler {
    fn start_profiling(&mut self, benchmark_id: &str, _benchmark_path: &Path) {
        assert!(benchmark_id.contains("profile_test"));
        self.started.set(self.started.get() + 1);
    }

    fn stop_profiling(&mut self, benchmark_id: &str, _benchmark_path: &Path) {
        assert!(benchmark_id.contains("profile_test"));
        self.stopped.set(self.stopped.get() + 1);
    }
}

// Verify that profilers are started and stopped as expected
#[test]
fn test_profiler_called() {
    let started = Rc::new(Cell::new(0u32));
    let stopped = Rc::new(Cell::new(0u32));
    let profiler = TestProfiler { started: started.clone(), stopped: stopped.clone() };
    let dir = temp_dir();
    let mut criterion =
        short_benchmark(&dir).with_profiler(profiler).profile_time(Some(Duration::from_secs(1)));
    criterion.bench_function("profile_test", |b| b.iter(|| 10));
    assert_eq!(1, started.get());
    assert_eq!(1, stopped.get());
}
