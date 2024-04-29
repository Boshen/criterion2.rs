//! A statistics-driven micro-benchmarking library written in Rust.
//!
//! This crate is a microbenchmarking library which aims to provide strong
//! statistical confidence in detecting and estimating the size of performance
//! improvements and regressions, while also being easy to use.
//!
//! See
//! [the user guide](https://bheisler.github.io/criterion.rs/book/index.html)
//! for examples as well as details on the measurement and analysis process,
//! and the output.
//!
//! ## Features:
//! * Collects detailed statistics, providing strong confidence that changes
//!   to performance are real, not measurement noise.
//! * Produces detailed charts, providing thorough understanding of your code's
//!   performance behavior.
//!
//! ## Feature flags
#![cfg_attr(feature = "document-features", doc = document_features::document_features!())]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
//!

#![allow(clippy::style, clippy::complexity)]
#![warn(bare_trait_objects)]
#![cfg_attr(feature = "codspeed", allow(unused))]

#[cfg(all(feature = "rayon", target_arch = "wasm32"))]
compile_error!("Rayon cannot be used when targeting wasi32. Try disabling default features.");

use serde::{Deserialize, Serialize};

// Needs to be declared before other modules
// in order to be usable there.
#[macro_use]
mod macros_private;
#[macro_use]
mod analysis;
mod benchmark;
#[macro_use]
mod benchmark_group;
#[cfg(feature = "codspeed")]
#[macro_use]
pub mod codspeed;
pub mod async_executor;
mod bencher;
mod cli;
mod connection;
mod criterion;
pub mod criterion_plot;
#[cfg(feature = "csv_output")]
mod csv_report;
mod error;
mod estimate;
mod format;
mod fs;
mod html;
mod kde;
pub mod measurement;
mod plot;
pub mod profiler;
mod report;
mod routine;
mod stats;

#[cfg(not(feature = "codspeed"))]
#[macro_use]
mod macros;
#[cfg(feature = "codspeed")]
#[macro_use]
mod macros_codspeed;

use std::default::Default;
use std::env;
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

use crate::criterion_plot::{Version, VersionError};

use crate::benchmark::BenchmarkConfig;
use crate::connection::Connection;
use crate::connection::OutgoingMessage;
use crate::html::Html;
use crate::measurement::{Measurement, WallTime};
#[cfg(feature = "plotters")]
use crate::plot::PlottersBackend;
use crate::plot::{Gnuplot, Plotter};
use crate::profiler::{ExternalProfiler, Profiler};
use crate::report::{BencherReport, CliReport, CliVerbosity, Report, ReportContext, Reports};

#[cfg(feature = "async")]
#[cfg(not(feature = "codspeed"))]
pub use crate::bencher::AsyncBencher;
#[cfg(feature = "async")]
#[cfg(feature = "codspeed")]
pub use crate::codspeed::bencher::AsyncBencher;

#[cfg(not(feature = "codspeed"))]
pub use crate::bencher::Bencher;
#[cfg(feature = "codspeed")]
pub use crate::codspeed::bencher::Bencher;

#[cfg(not(feature = "codspeed"))]
pub use crate::benchmark_group::{BenchmarkGroup, BenchmarkId};
#[cfg(feature = "codspeed")]
pub use crate::codspeed::benchmark_group::{BenchmarkGroup, BenchmarkId};

#[cfg(feature = "codspeed")]
pub use crate::codspeed::criterion::Criterion;
#[cfg(not(feature = "codspeed"))]
pub use crate::criterion::Criterion;

fn gnuplot_version() -> &'static Result<Version, VersionError> {
    static GNUPLOT_VERSION: OnceLock<Result<Version, VersionError>> = OnceLock::new();

    GNUPLOT_VERSION.get_or_init(criterion_plot::version)
}

fn default_plotting_backend() -> &'static PlottingBackend {
    static DEFAULT_PLOTTING_BACKEND: OnceLock<PlottingBackend> = OnceLock::new();

    DEFAULT_PLOTTING_BACKEND.get_or_init(|| match gnuplot_version() {
        Ok(_) => PlottingBackend::Gnuplot,
        #[cfg(feature = "plotters")]
        Err(e) => {
            match e {
                VersionError::Exec(_) => eprintln!("Gnuplot not found, using plotters backend"),
                e => eprintln!("Gnuplot not found or not usable, using plotters backend\n{}", e),
            };
            PlottingBackend::Plotters
        }
        #[cfg(not(feature = "plotters"))]
        Err(_) => PlottingBackend::None,
    })
}

fn cargo_criterion_connection() -> &'static Option<Mutex<Connection>> {
    static CARGO_CRITERION_CONNECTION: OnceLock<Option<Mutex<Connection>>> = OnceLock::new();

    CARGO_CRITERION_CONNECTION.get_or_init(|| match std::env::var("CARGO_CRITERION_PORT") {
        Ok(port_str) => {
            let port: u16 = port_str.parse().ok()?;
            let stream = TcpStream::connect(("localhost", port)).ok()?;
            Some(Mutex::new(Connection::new(stream).ok()?))
        }
        Err(_) => None,
    })
}

fn default_output_directory() -> &'static PathBuf {
    static DEFAULT_OUTPUT_DIRECTORY: OnceLock<PathBuf> = OnceLock::new();

    DEFAULT_OUTPUT_DIRECTORY.get_or_init(|| {
        // Set criterion home to (in descending order of preference):
        // - $CRITERION_HOME (cargo-criterion sets this, but other users could as well)
        // - $CARGO_TARGET_DIR/criterion
        // - the cargo target dir from `cargo metadata`
        // - ./target/criterion
        if let Some(value) = env::var_os("CRITERION_HOME") {
            PathBuf::from(value)
        } else if let Some(path) = cargo_target_directory() {
            path.join("criterion")
        } else {
            PathBuf::from("target/criterion")
        }
    })
}

fn debug_enabled() -> bool {
    static DEBUG_ENABLED: OnceLock<bool> = OnceLock::new();

    *DEBUG_ENABLED.get_or_init(|| std::env::var_os("CRITERION_DEBUG").is_some())
}

/// Reexport of [std::hint::black_box].
#[inline]
pub fn black_box<T>(dummy: T) -> T {
    std::hint::black_box(dummy)
}

/// Argument to [`Bencher::iter_batched`] and [`Bencher::iter_batched_ref`] which controls the
/// batch size.
///
/// Generally speaking, almost all benchmarks should use `SmallInput`. If the input or the result
/// of the benchmark routine is large enough that `SmallInput` causes out-of-memory errors,
/// `LargeInput` can be used to reduce memory usage at the cost of increasing the measurement
/// overhead. If the input or the result is extremely large (or if it holds some
/// limited external resource like a file handle), `PerIteration` will set the number of iterations
/// per batch to exactly one. `PerIteration` can increase the measurement overhead substantially
/// and should be avoided wherever possible.
///
/// Each value lists an estimate of the measurement overhead. This is intended as a rough guide
/// to assist in choosing an option, it should not be relied upon. In particular, it is not valid
/// to subtract the listed overhead from the measurement and assume that the result represents the
/// true runtime of a function. The actual measurement overhead for your specific benchmark depends
/// on the details of the function you're benchmarking and the hardware and operating
/// system running the benchmark.
///
/// With that said, if the runtime of your function is small relative to the measurement overhead
/// it will be difficult to take accurate measurements. In this situation, the best option is to use
/// [`Bencher::iter`] which has next-to-zero measurement overhead.
#[derive(Debug, Eq, PartialEq, Copy, Hash, Clone)]
pub enum BatchSize {
    /// `SmallInput` indicates that the input to the benchmark routine (the value returned from
    /// the setup routine) is small enough that millions of values can be safely held in memory.
    /// Always prefer `SmallInput` unless the benchmark is using too much memory.
    ///
    /// In testing, the maximum measurement overhead from benchmarking with `SmallInput` is on the
    /// order of 500 picoseconds. This is presented as a rough guide; your results may vary.
    SmallInput,

    /// `LargeInput` indicates that the input to the benchmark routine or the value returned from
    /// that routine is large. This will reduce the memory usage but increase the measurement
    /// overhead.
    ///
    /// In testing, the maximum measurement overhead from benchmarking with `LargeInput` is on the
    /// order of 750 picoseconds. This is presented as a rough guide; your results may vary.
    LargeInput,

    /// `PerIteration` indicates that the input to the benchmark routine or the value returned from
    /// that routine is extremely large or holds some limited resource, such that holding many values
    /// in memory at once is infeasible. This provides the worst measurement overhead, but the
    /// lowest memory usage.
    ///
    /// In testing, the maximum measurement overhead from benchmarking with `PerIteration` is on the
    /// order of 350 nanoseconds or 350,000 picoseconds. This is presented as a rough guide; your
    /// results may vary.
    PerIteration,

    /// `NumBatches` will attempt to divide the iterations up into a given number of batches.
    /// A larger number of batches (and thus smaller batches) will reduce memory usage but increase
    /// measurement overhead. This allows the user to choose their own tradeoff between memory usage
    /// and measurement overhead, but care must be taken in tuning the number of batches. Most
    /// benchmarks should use `SmallInput` or `LargeInput` instead.
    NumBatches(u64),

    /// `NumIterations` fixes the batch size to a constant number, specified by the user. This
    /// allows the user to choose their own tradeoff between overhead and memory usage, but care must
    /// be taken in tuning the batch size. In general, the measurement overhead of `NumIterations`
    /// will be larger than that of `NumBatches`. Most benchmarks should use `SmallInput` or
    /// `LargeInput` instead.
    NumIterations(u64),

    #[doc(hidden)]
    __NonExhaustive,
}
impl BatchSize {
    /// Convert to a number of iterations per batch.
    ///
    /// We try to do a constant number of batches regardless of the number of iterations in this
    /// sample. If the measurement overhead is roughly constant regardless of the number of
    /// iterations the analysis of the results later will have an easier time separating the
    /// measurement overhead from the benchmark time.
    fn iters_per_batch(self, iters: u64) -> u64 {
        match self {
            BatchSize::SmallInput => (iters + 10 - 1) / 10,
            BatchSize::LargeInput => (iters + 1000 - 1) / 1000,
            BatchSize::PerIteration => 1,
            BatchSize::NumBatches(batches) => (iters + batches - 1) / batches,
            BatchSize::NumIterations(size) => size,
            BatchSize::__NonExhaustive => panic!("__NonExhaustive is not a valid BatchSize."),
        }
    }
}

/// Baseline describes how the `baseline_directory` is handled.
#[derive(Debug, Clone, Copy)]
pub enum Baseline {
    /// `CompareLenient` compares against a previous saved version of the baseline.
    /// If a previous baseline does not exist, the benchmark is run as normal but no comparison occurs.
    CompareLenient,
    /// `CompareStrict` compares against a previous saved version of the baseline.
    /// If a previous baseline does not exist, a panic occurs.
    CompareStrict,
    /// `Save` writes the benchmark results to the baseline directory,
    /// overwriting any results that were previously there.
    Save,
    /// `Discard` benchmark results.
    Discard,
}

/// Enum used to select the plotting backend.
#[derive(Debug, Clone, Copy)]
pub enum PlottingBackend {
    /// Plotting backend which uses the external `gnuplot` command to render plots. This is the
    /// default if the `gnuplot` command is installed.
    Gnuplot,
    /// Plotting backend which uses the rust 'Plotters' library. This is the default if `gnuplot`
    /// is not installed.
    Plotters,
    /// Null plotting backend which outputs nothing,
    None,
}
impl PlottingBackend {
    fn create_plotter(&self) -> Option<Box<dyn Plotter>> {
        match self {
            PlottingBackend::Gnuplot => Some(Box::<Gnuplot>::default()),
            #[cfg(feature = "plotters")]
            PlottingBackend::Plotters => Some(Box::<PlottersBackend>::default()),
            #[cfg(not(feature = "plotters"))]
            PlottingBackend::Plotters => panic!("Criterion was built without plotters support."),
            PlottingBackend::None => None,
        }
    }
}
impl std::str::FromStr for PlottingBackend {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "gnuplot" => Self::Gnuplot,
            "plotters" => Self::Plotters,
            "none" => Self::None,
            _ => return Err("Valid values are gnuplot, plotters and none"),
        })
    }
}

impl std::fmt::Display for PlottingBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PlottingBackend::Gnuplot => "gnuplot",
            PlottingBackend::Plotters => "plotters",
            PlottingBackend::None => "none",
        })
    }
}

#[derive(Debug, Clone)]
/// Enum representing the execution mode.
pub(crate) enum Mode {
    /// Run benchmarks normally.
    Benchmark,
    /// List all benchmarks but do not run them.
    List(ListFormat),
    /// Run benchmarks once to verify that they work, but otherwise do not measure them.
    Test,
    /// Iterate benchmarks for a given length of time but do not analyze or report on them.
    Profile(Duration),
}
impl Mode {
    pub fn is_benchmark(&self) -> bool {
        matches!(self, Mode::Benchmark)
    }

    pub fn is_terse(&self) -> bool {
        matches!(self, Mode::List(ListFormat::Terse))
    }
}

#[derive(Debug, Clone, Copy)]
/// Enum representing the list format.
pub(crate) enum ListFormat {
    /// The regular, default format.
    Pretty,
    /// The terse format, where nothing other than the name of the test and ": benchmark" at the end
    /// is printed out.
    Terse,
}

impl Default for ListFormat {
    fn default() -> Self {
        Self::Pretty
    }
}

/// Benchmark filtering support.
#[derive(Clone, Debug)]
pub enum BenchmarkFilter {
    /// Run all benchmarks.
    AcceptAll,
    /// Run the benchmark matching this string exactly.
    Exact(String),
    /// Do not run any benchmarks.
    RejectAll,
}

/// Returns the Cargo target directory, possibly calling `cargo metadata` to
/// figure it out.
fn cargo_target_directory() -> Option<PathBuf> {
    #[derive(Deserialize)]
    struct Metadata {
        target_directory: PathBuf,
    }

    env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).or_else(|| {
        let output = Command::new(env::var_os("CARGO")?)
            .args(["metadata", "--format-version", "1"])
            .output()
            .ok()?;
        let metadata: Metadata = serde_json::from_slice(&output.stdout).ok()?;
        Some(metadata.target_directory)
    })
}

/// Enum representing different ways of measuring the throughput of benchmarked code.
/// If the throughput setting is configured for a benchmark then the estimated throughput will
/// be reported as well as the time per iteration.
// TODO: Remove serialize/deserialize from the public API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Throughput {
    /// Measure throughput in terms of bytes/second. The value should be the number of bytes
    /// processed by one iteration of the benchmarked code. Typically, this would be the length of
    /// an input string or `&[u8]`.
    Bytes(u64),

    /// Equivalent to Bytes, but the value will be reported in terms of
    /// kilobytes (1000 bytes) per second instead of kibibytes (1024 bytes) per
    /// second, megabytes instead of mibibytes, and gigabytes instead of gibibytes.
    BytesDecimal(u64),

    /// Measure throughput in terms of elements/second. The value should be the number of elements
    /// processed by one iteration of the benchmarked code. Typically, this would be the size of a
    /// collection, but could also be the number of lines of input text or the number of values to
    /// parse.
    Elements(u64),
}

/// Axis scaling type
#[derive(Debug, Clone, Copy)]
pub enum AxisScale {
    /// Axes scale linearly
    Linear,

    /// Axes scale logarithmically
    Logarithmic,
}

/// Contains the configuration options for the plots generated by a particular benchmark
/// or benchmark group.
///
/// ```rust
/// use self::criterion2::{Bencher, Criterion, PlotConfiguration, AxisScale};
///
/// let plot_config = PlotConfiguration::default()
///     .summary_scale(AxisScale::Logarithmic);
///
/// // Using Criterion::default() for simplicity; normally you'd use the macros.
/// let mut criterion = Criterion::default();
/// let mut benchmark_group = criterion.benchmark_group("Group name");
/// benchmark_group.plot_config(plot_config);
/// // Use benchmark group
/// ```
#[derive(Debug, Clone)]
pub struct PlotConfiguration {
    summary_scale: AxisScale,
}

impl Default for PlotConfiguration {
    fn default() -> PlotConfiguration {
        PlotConfiguration { summary_scale: AxisScale::Linear }
    }
}

impl PlotConfiguration {
    #[must_use]
    /// Set the axis scale (linear or logarithmic) for the summary plots. Typically, you would
    /// set this to logarithmic if benchmarking over a range of inputs which scale exponentially.
    /// Defaults to linear.
    pub fn summary_scale(mut self, new_scale: AxisScale) -> PlotConfiguration {
        self.summary_scale = new_scale;
        self
    }
}

/// This enum allows the user to control how Criterion.rs chooses the iteration count when sampling.
/// The default is Auto, which will choose a method automatically based on the iteration time during
/// the warm-up phase.
#[derive(Debug, Clone, Copy)]
pub enum SamplingMode {
    /// Criterion.rs should choose a sampling method automatically. This is the default, and is
    /// recommended for most users and most benchmarks.
    Auto,

    /// Scale the iteration count in each sample linearly. This is suitable for most benchmarks,
    /// but it tends to require many iterations which can make it very slow for very long benchmarks.
    Linear,

    /// Keep the iteration count the same for all samples. This is not recommended, as it affects
    /// the statistics that Criterion.rs can compute. However, it requires fewer iterations than
    /// the Linear method and therefore is more suitable for very long-running benchmarks where
    /// benchmark execution time is more of a problem and statistical precision is less important.
    Flat,
}
impl SamplingMode {
    pub(crate) fn choose_sampling_mode(
        &self,
        warmup_mean_execution_time: f64,
        sample_count: u64,
        target_time: f64,
    ) -> ActualSamplingMode {
        match self {
            SamplingMode::Linear => ActualSamplingMode::Linear,
            SamplingMode::Flat => ActualSamplingMode::Flat,
            SamplingMode::Auto => {
                // Estimate execution time with linear sampling
                let total_runs = sample_count * (sample_count + 1) / 2;
                let d =
                    (target_time / warmup_mean_execution_time / total_runs as f64).ceil() as u64;
                let expected_ns = total_runs as f64 * d as f64 * warmup_mean_execution_time;

                if expected_ns > (2.0 * target_time) {
                    ActualSamplingMode::Flat
                } else {
                    ActualSamplingMode::Linear
                }
            }
        }
    }
}

/// Enum to represent the sampling mode without Auto.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) enum ActualSamplingMode {
    Linear,
    Flat,
}
impl ActualSamplingMode {
    pub(crate) fn iteration_counts(
        &self,
        warmup_mean_execution_time: f64,
        sample_count: u64,
        target_time: &Duration,
    ) -> Vec<u64> {
        match self {
            ActualSamplingMode::Linear => {
                let n = sample_count;
                let met = warmup_mean_execution_time;
                let m_ns = target_time.as_nanos();
                // Solve: [d + 2*d + 3*d + ... + n*d] * met = m_ns
                let total_runs = n * (n + 1) / 2;
                let d = ((m_ns as f64 / met / total_runs as f64).ceil() as u64).max(1);
                let expected_ns = total_runs as f64 * d as f64 * met;

                if d == 1 {
                    let recommended_sample_size =
                        ActualSamplingMode::recommend_linear_sample_size(m_ns as f64, met);
                    let actual_time = Duration::from_nanos(expected_ns as u64);
                    eprint!("\nWarning: Unable to complete {} samples in {:.1?}. You may wish to increase target time to {:.1?}",
                            n, target_time, actual_time);

                    if recommended_sample_size != n {
                        eprintln!(
                            ", enable flat sampling, or reduce sample count to {}.",
                            recommended_sample_size
                        );
                    } else {
                        eprintln!(" or enable flat sampling.");
                    }
                }

                (1..(n + 1)).map(|a| a * d).collect::<Vec<u64>>()
            }
            ActualSamplingMode::Flat => {
                let n = sample_count;
                let met = warmup_mean_execution_time;
                let m_ns = target_time.as_nanos() as f64;
                let time_per_sample = m_ns / (n as f64);
                // This is pretty simplistic; we could do something smarter to fit into the allotted time.
                let iterations_per_sample = ((time_per_sample / met).ceil() as u64).max(1);

                let expected_ns = met * (iterations_per_sample * n) as f64;

                if iterations_per_sample == 1 {
                    let recommended_sample_size =
                        ActualSamplingMode::recommend_flat_sample_size(m_ns, met);
                    let actual_time = Duration::from_nanos(expected_ns as u64);
                    eprint!("\nWarning: Unable to complete {} samples in {:.1?}. You may wish to increase target time to {:.1?}",
                            n, target_time, actual_time);

                    if recommended_sample_size != n {
                        eprintln!(", or reduce sample count to {}.", recommended_sample_size);
                    } else {
                        eprintln!(".");
                    }
                }

                vec![iterations_per_sample; n as usize]
            }
        }
    }

    fn is_linear(&self) -> bool {
        matches!(self, ActualSamplingMode::Linear)
    }

    fn recommend_linear_sample_size(target_time: f64, met: f64) -> u64 {
        // Some math shows that n(n+1)/2 * d * met = target_time. d = 1, so it can be ignored.
        // This leaves n(n+1) = (2*target_time)/met, or n^2 + n - (2*target_time)/met = 0
        // Which can be solved with the quadratic formula. Since A and B are constant 1,
        // this simplifies to sample_size = (-1 +- sqrt(1 - 4C))/2, where C = (2*target_time)/met.
        // We don't care about the negative solution. Experimentation shows that this actually tends to
        // result in twice the desired execution time (probably because of the ceil used to calculate
        // d) so instead I use c = target_time/met.
        let c = target_time / met;
        let sample_size = (-1.0 + (4.0 * c).sqrt()) / 2.0;
        let sample_size = sample_size as u64;

        // Round down to the nearest 10 to give a margin and avoid excessive precision
        let sample_size = (sample_size / 10) * 10;

        // Clamp it to be at least 10, since criterion.rs doesn't allow sample sizes smaller than 10.
        if sample_size < 10 {
            10
        } else {
            sample_size
        }
    }

    fn recommend_flat_sample_size(target_time: f64, met: f64) -> u64 {
        let sample_size = (target_time / met) as u64;

        // Round down to the nearest 10 to give a margin and avoid excessive precision
        let sample_size = (sample_size / 10) * 10;

        // Clamp it to be at least 10, since criterion.rs doesn't allow sample sizes smaller than 10.
        if sample_size < 10 {
            10
        } else {
            sample_size
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SavedSample {
    sampling_mode: ActualSamplingMode,
    iters: Vec<f64>,
    times: Vec<f64>,
}

/// Custom-test-framework runner. Should not be called directly.
#[doc(hidden)]
pub fn runner(benches: &[&dyn Fn()]) {
    for bench in benches {
        bench();
    }
    crate::criterion::Criterion::default().configure_from_args().final_summary();
}
