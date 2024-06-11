use std::{
    cell::RefCell,
    collections::HashSet,
    io::{stdout, IsTerminal},
    path::{Path, PathBuf},
    sync::MutexGuard,
    time::Duration,
};

use crate::{
    bencher::Bencher,
    benchmark_group::{BenchmarkGroup, BenchmarkId},
    cargo_criterion_connection, debug_enabled, default_output_directory, Baseline, BencherReport,
    BenchmarkConfig, BenchmarkFilter, CliReport, CliVerbosity, Connection, ExternalProfiler,
    Measurement, Mode, OutgoingMessage, Profiler, Report, ReportContext, Reports, WallTime,
};

/// The benchmark manager
///
/// `Criterion` lets you configure and execute benchmarks
///
/// Each benchmark consists of four phases:
///
/// - **Warm-up**: The routine is repeatedly executed, to let the CPU/OS/JIT/interpreter adapt to
/// the new load
/// - **Measurement**: The routine is repeatedly executed, and timing information is collected into
/// a sample
/// - **Analysis**: The sample is analyzed and distilled into meaningful statistics that get
/// reported to stdout, stored in files, and plotted
/// - **Comparison**: The current sample is compared with the sample obtained in the previous
/// benchmark.
pub struct Criterion<M: Measurement = WallTime> {
    pub(crate) config: BenchmarkConfig,
    pub(crate) filter: BenchmarkFilter,
    pub(crate) report: Reports,
    pub(crate) output_directory: PathBuf,
    pub(crate) baseline_directory: String,
    pub(crate) baseline: Baseline,
    pub(crate) load_baseline: Option<String>,
    pub(crate) all_directories: HashSet<String>,
    pub(crate) all_titles: HashSet<String>,
    pub(crate) measurement: M,
    pub(crate) profiler: Box<RefCell<dyn Profiler>>,
    pub(crate) connection: Option<MutexGuard<'static, Connection>>,
    pub(crate) mode: Mode,
}

impl Default for Criterion {
    /// Creates a benchmark manager with the following default settings:
    ///
    /// - Sample size: 100 measurements
    /// - Warm-up time: 3 s
    /// - Measurement time: 5 s
    /// - Bootstrap size: 100 000 resamples
    /// - Noise threshold: 0.01 (1%)
    /// - Confidence level: 0.95
    /// - Significance level: 0.05
    /// - Plotting: enabled, using gnuplot if available or plotters if gnuplot is not available
    /// - No filter
    fn default() -> Criterion {
        let reports = Reports {
            cli_enabled: true,
            cli: CliReport::new(false, false, CliVerbosity::Normal),
            bencher_enabled: false,
            bencher: BencherReport,
        };

        let mut criterion = Criterion {
            config: BenchmarkConfig::default(),
            filter: BenchmarkFilter::AcceptAll,
            report: reports,
            baseline_directory: "base".to_owned(),
            baseline: Baseline::Save,
            load_baseline: None,
            output_directory: default_output_directory().clone(),
            all_directories: HashSet::new(),
            all_titles: HashSet::new(),
            measurement: WallTime,
            profiler: Box::new(RefCell::new(ExternalProfiler)),
            connection: cargo_criterion_connection().as_ref().map(|mtx| mtx.lock().unwrap()),
            mode: Mode::Benchmark,
        };

        if criterion.connection.is_some() {
            // disable all reports when connected to cargo-criterion; it will do the reporting.
            criterion.report.cli_enabled = false;
            criterion.report.bencher_enabled = false;
        }
        criterion
    }
}

impl<M: Measurement> Criterion<M> {
    /// Changes the measurement for the benchmarks run with this runner. See the
    /// Measurement trait for more details
    pub fn with_measurement<M2: Measurement>(self, m: M2) -> Criterion<M2> {
        // Can't use struct update syntax here because they're technically different types.
        Criterion {
            config: self.config,
            filter: self.filter,
            report: self.report,
            baseline_directory: self.baseline_directory,
            baseline: self.baseline,
            load_baseline: self.load_baseline,
            output_directory: self.output_directory,
            all_directories: self.all_directories,
            all_titles: self.all_titles,
            measurement: m,
            profiler: self.profiler,
            connection: self.connection,
            mode: self.mode,
        }
    }

    #[must_use]
    /// Changes the internal profiler for benchmarks run with this runner. See
    /// the Profiler trait for more details.
    pub fn with_profiler<P: Profiler + 'static>(self, p: P) -> Criterion<M> {
        Criterion { profiler: Box::new(RefCell::new(p)), ..self }
    }

    #[must_use]
    /// Changes the default size of the sample for benchmarks run with this runner.
    ///
    /// A bigger sample should yield more accurate results if paired with a sufficiently large
    /// measurement time.
    ///
    /// Sample size must be at least 10.
    ///
    /// # Panics
    ///
    /// Panics if n < 10
    pub fn sample_size(mut self, n: usize) -> Criterion<M> {
        assert!(n >= 10);

        self.config.sample_size = n;
        self
    }

    #[must_use]
    /// Changes the default warm up time for benchmarks run with this runner.
    ///
    /// # Panics
    ///
    /// Panics if the input duration is zero
    pub fn warm_up_time(mut self, dur: Duration) -> Criterion<M> {
        assert!(dur.as_nanos() > 0);

        self.config.warm_up_time = dur;
        self
    }

    #[must_use]
    /// Changes the default measurement time for benchmarks run with this runner.
    ///
    /// With a longer time, the measurement will become more resilient to transitory peak loads
    /// caused by external programs
    ///
    /// **Note**: If the measurement time is too "low", Criterion will automatically increase it
    ///
    /// # Panics
    ///
    /// Panics if the input duration in zero
    pub fn measurement_time(mut self, dur: Duration) -> Criterion<M> {
        assert!(dur.as_nanos() > 0);

        self.config.measurement_time = dur;
        self
    }

    #[must_use]
    /// Changes the default number of resamples for benchmarks run with this runner.
    ///
    /// Number of resamples to use for the
    /// [bootstrap](http://en.wikipedia.org/wiki/Bootstrapping_(statistics)#Case_resampling)
    ///
    /// A larger number of resamples reduces the random sampling errors, which are inherent to the
    /// bootstrap method, but also increases the analysis time
    ///
    /// # Panics
    ///
    /// Panics if the number of resamples is set to zero
    pub fn nresamples(mut self, n: usize) -> Criterion<M> {
        assert!(n > 0);
        if n <= 1000 {
            eprintln!("\nWarning: It is not recommended to reduce nresamples below 1000.");
        }

        self.config.nresamples = n;
        self
    }

    #[must_use]
    /// Changes the default noise threshold for benchmarks run with this runner. The noise threshold
    /// is used to filter out small changes in performance, even if they are statistically
    /// significant. Sometimes benchmarking the same code twice will result in small but
    /// statistically significant differences solely because of noise. This provides a way to filter
    /// out some of these false positives at the cost of making it harder to detect small changes
    /// to the true performance of the benchmark.
    ///
    /// The default is 0.01, meaning that changes smaller than 1% will be ignored.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is set to a negative value
    pub fn noise_threshold(mut self, threshold: f64) -> Criterion<M> {
        assert!(threshold >= 0.0);

        self.config.noise_threshold = threshold;
        self
    }

    #[must_use]
    /// Changes the default confidence level for benchmarks run with this runner. The confidence
    /// level is the desired probability that the true runtime lies within the estimated
    /// [confidence interval](https://en.wikipedia.org/wiki/Confidence_interval). The default is
    /// 0.95, meaning that the confidence interval should capture the true value 95% of the time.
    ///
    /// # Panics
    ///
    /// Panics if the confidence level is set to a value outside the `(0, 1)` range
    pub fn confidence_level(mut self, cl: f64) -> Criterion<M> {
        assert!(cl > 0.0 && cl < 1.0);
        if cl < 0.5 {
            eprintln!("\nWarning: It is not recommended to reduce confidence level below 0.5.");
        }

        self.config.confidence_level = cl;
        self
    }

    #[must_use]
    /// Changes the default [significance level](https://en.wikipedia.org/wiki/Statistical_significance)
    /// for benchmarks run with this runner. This is used to perform a
    /// [hypothesis test](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing) to see if
    /// the measurements from this run are different from the measured performance of the last run.
    /// The significance level is the desired probability that two measurements of identical code
    /// will be considered 'different' due to noise in the measurements. The default value is 0.05,
    /// meaning that approximately 5% of identical benchmarks will register as different due to
    /// noise.
    ///
    /// This presents a trade-off. By setting the significance level closer to 0.0, you can increase
    /// the statistical robustness against noise, but it also weakens Criterion.rs' ability to
    /// detect small but real changes in the performance. By setting the significance level
    /// closer to 1.0, Criterion.rs will be more able to detect small true changes, but will also
    /// report more spurious differences.
    ///
    /// See also the noise threshold setting.
    ///
    /// # Panics
    ///
    /// Panics if the significance level is set to a value outside the `(0, 1)` range
    pub fn significance_level(mut self, sl: f64) -> Criterion<M> {
        assert!(sl > 0.0 && sl < 1.0);

        self.config.significance_level = sl;
        self
    }

    #[must_use]
    /// Names an explicit baseline and enables overwriting the previous results.
    pub fn save_baseline(mut self, baseline: String) -> Criterion<M> {
        self.baseline_directory = baseline;
        self.baseline = Baseline::Save;
        self
    }

    #[must_use]
    /// Names an explicit baseline and disables overwriting the previous results.
    pub fn retain_baseline(mut self, baseline: String, strict: bool) -> Criterion<M> {
        self.baseline_directory = baseline;
        self.baseline = if strict { Baseline::CompareStrict } else { Baseline::CompareLenient };
        self
    }

    /// Only run benchmarks specified by the given filter.
    ///
    /// This overwrites [`Self::with_filter`].
    pub fn with_benchmark_filter(mut self, filter: BenchmarkFilter) -> Criterion<M> {
        self.filter = filter;

        self
    }

    #[must_use]
    /// Override whether the CLI output will be colored or not. Usually you would use the `--color`
    /// CLI argument, but this is available for programmmatic use as well.
    pub fn with_output_color(mut self, enabled: bool) -> Criterion<M> {
        self.report.cli.enable_text_coloring = enabled;
        self
    }

    /// Set the output directory (currently for testing only)
    #[must_use]
    #[doc(hidden)]
    pub fn output_directory(mut self, path: &Path) -> Criterion<M> {
        path.clone_into(&mut self.output_directory);

        self
    }

    /// Set the profile time (currently for testing only)
    #[must_use]
    #[doc(hidden)]
    pub fn profile_time(mut self, profile_time: Option<Duration>) -> Criterion<M> {
        match profile_time {
            Some(time) => self.mode = Mode::Profile(time),
            None => self.mode = Mode::Benchmark,
        }

        self
    }

    /// Generate the final summary at the end of a run.
    #[doc(hidden)]
    pub fn final_summary(&self) {
        if !self.mode.is_benchmark() {
            return;
        }

        let report_context = ReportContext { output_directory: self.output_directory.clone() };

        self.report.final_summary(&report_context);
    }

    /// Configure this criterion struct based on the command-line arguments to
    /// this process.
    #[must_use]
    pub fn configure_from_args(mut self) -> Criterion<M> {
        use crate::cli::*;

        let opts = options(&self.config).fallback_to_usage().run();

        if self.connection.is_some() {
            if opts.color != Color::Auto {
                eprintln!(
                    "Warning: --color will be ignored when running with cargo-criterion. Use `cargo criterion --color {} -- <args>` instead.",
                    opts.color
                );
            }

            // What about quiet?
            if opts.verbosity == CliVerbosity::Verbose {
                eprintln!(
                    "Warning: --verbose will be ignored when running with cargo-criterion. Use `cargo criterion --output-format verbose -- <args>` instead."
                );
            }
            if opts.output_format != OutputFormat::Criterion {
                eprintln!(
                    "Warning: --output-format will be ignored when running with cargo-criterion. Use `cargo criterion --output-format {} -- <args>` instead.",
                    opts.output_format
                );
            }

            // TODO - currently baseline stuff seem to be partially coupled with operations
            if matches!(opts.op, Op::LoadBaseline(_)) {
                eprintln!("Error: baselines are not supported when running with cargo-criterion.");
                std::process::exit(1);
            }
        }

        self.mode = match opts.op {
            Op::List => Mode::List(opts.format),
            Op::LoadBaseline(ref dir) => {
                self.load_baseline = Some(dir.to_owned());
                Mode::Benchmark
            }
            Op::ProfileTime(t) => Mode::Profile(t),
            Op::Test => Mode::Test,
            Op::Benchmark => Mode::Benchmark,
        };

        // This is kind of a hack, but disable the connection to the runner if we're not benchmarking.
        if !self.mode.is_benchmark() {
            self.connection = None;
        }

        let filter = if opts.ignored {
            // --ignored overwrites any name-based filters passed in.
            BenchmarkFilter::RejectAll
        } else if let Some(filter) = opts.filter.as_ref() {
            // if opts.exact {
            BenchmarkFilter::Exact(filter.to_owned())
            // }
        } else {
            BenchmarkFilter::AcceptAll
        };
        self = self.with_benchmark_filter(filter);

        match opts.baseline {
            Baseline_::Save(ref dir) => {
                self.baseline = Baseline::Save;
                dir.clone_into(&mut self.baseline_directory)
            }
            Baseline_::Discard => {
                self.baseline = Baseline::Discard;
            }
            Baseline_::Lenient(ref dir) => {
                self.baseline = Baseline::CompareLenient;
                dir.clone_into(&mut self.baseline_directory);
            }
            Baseline_::Strict(ref dir) => {
                self.baseline = Baseline::CompareStrict;
                dir.clone_into(&mut self.baseline_directory);
            }
        }

        if self.connection.is_some() {
            // disable all reports when connected to cargo-criterion; it will do the reporting.
            self.report.cli_enabled = false;
            self.report.bencher_enabled = false;
        } else {
            match opts.output_format {
                OutputFormat::Bencher => {
                    self.report.bencher_enabled = true;
                    self.report.cli_enabled = false;
                }
                OutputFormat::Criterion => {
                    let verbosity = opts.verbosity;
                    let verbose = opts.verbosity == CliVerbosity::Verbose;

                    let stdout_isatty = stdout().is_terminal();
                    let mut enable_text_overwrite = stdout_isatty && !verbose && !debug_enabled();
                    let enable_text_coloring;
                    match opts.color {
                        Color::Auto => enable_text_coloring = stdout_isatty,
                        Color::Always => enable_text_coloring = true,
                        Color::Never => {
                            enable_text_coloring = false;
                            enable_text_overwrite = false;
                        }
                    }
                    self.report.bencher_enabled = false;
                    self.report.cli_enabled = true;
                    self.report.cli = CliReport::new(
                        enable_text_overwrite,
                        enable_text_coloring,
                        verbosity.into(),
                    );
                }
            }
        }

        match opts.sample {
            Sample::Specific(size) => {
                assert!(size >= 10);
                self.config.sample_size = size;
            }
            Sample::Quick => self.config.quick_mode = true,
        }

        assert!(opts.warm_up_time > Duration::from_secs(0));
        self.config.warm_up_time = opts.warm_up_time;

        assert!(opts.measurement_time > Duration::from_secs(0));
        self.config.measurement_time = opts.measurement_time;

        assert!(opts.nresamples > 0);
        self.config.nresamples = opts.nresamples;

        assert!(opts.noise_threshold > 0.0);
        self.config.noise_threshold = opts.noise_threshold;

        assert!(opts.confidence_level > 0.0 && opts.confidence_level < 1.0);
        self.config.confidence_level = opts.confidence_level;

        assert!(opts.significance_level > 0.0 && opts.significance_level < 1.0);
        self.config.significance_level = opts.significance_level;

        self
    }

    pub(crate) fn filter_matches(&self, id: &str) -> bool {
        match &self.filter {
            BenchmarkFilter::AcceptAll => true,
            BenchmarkFilter::Exact(exact) => id == exact,
            BenchmarkFilter::RejectAll => false,
        }
    }

    /// Returns true iff we should save the benchmark results in
    /// json files on the local disk.
    pub(crate) fn should_save_baseline(&self) -> bool {
        self.connection.is_none()
            && self.load_baseline.is_none()
            && !matches!(self.baseline, Baseline::Discard)
    }

    /// Return a benchmark group. All benchmarks performed using a benchmark group will be
    /// grouped together in the final report.
    ///
    /// # Examples:
    ///
    /// ```rust
    /// use self::criterion::*;
    ///
    /// fn bench_simple(c: &mut Criterion) {
    ///     let mut group = c.benchmark_group("My Group");
    ///
    ///     // Now we can perform benchmarks with this group
    ///     group.bench_function("Bench 1", |b| b.iter(|| 1 ));
    ///     group.bench_function("Bench 2", |b| b.iter(|| 2 ));
    ///
    ///     group.finish();
    /// }
    /// criterion_group!(benches, bench_simple);
    /// criterion_main!(benches);
    /// ```
    /// # Panics:
    /// Panics if the group name is empty
    pub fn benchmark_group<S: Into<String>>(&mut self, group_name: S) -> BenchmarkGroup<'_, M> {
        let group_name = group_name.into();
        assert!(!group_name.is_empty(), "Group name must not be empty.");

        if let Some(conn) = &self.connection {
            conn.send(&OutgoingMessage::BeginningBenchmarkGroup { group: &group_name }).unwrap();
        }

        BenchmarkGroup::new(self, group_name)
    }
}
impl<M> Criterion<M>
where
    M: Measurement + 'static,
{
    /// Benchmarks a function. For comparing multiple functions, see `benchmark_group`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use self::criterion::*;
    ///
    /// fn bench(c: &mut Criterion) {
    ///     // Setup (construct data, allocate memory, etc)
    ///     c.bench_function(
    ///         "function_name",
    ///         |b| b.iter(|| {
    ///             // Code to benchmark goes here
    ///         }),
    ///     );
    /// }
    ///
    /// criterion_group!(benches, bench);
    /// criterion_main!(benches);
    /// ```
    pub fn bench_function<F>(&mut self, id: &str, f: F) -> &mut Criterion<M>
    where
        F: FnMut(&mut Bencher<'_, M>),
    {
        self.benchmark_group(id).bench_function(BenchmarkId::no_function(), f);
        self
    }

    /// Benchmarks a function with an input. For comparing multiple functions or multiple inputs,
    /// see `benchmark_group`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use self::criterion::*;
    ///
    /// fn bench(c: &mut Criterion) {
    ///     // Setup (construct data, allocate memory, etc)
    ///     let input = 5u64;
    ///     c.bench_with_input(
    ///         BenchmarkId::new("function_name", input), &input,
    ///         |b, i| b.iter(|| {
    ///             // Code to benchmark using input `i` goes here
    ///         }),
    ///     );
    /// }
    ///
    /// criterion_group!(benches, bench);
    /// criterion_main!(benches);
    /// ```
    pub fn bench_with_input<F, I>(&mut self, id: BenchmarkId, input: &I, f: F) -> &mut Criterion<M>
    where
        F: FnMut(&mut Bencher<'_, M>, &I),
    {
        // It's possible to use BenchmarkId::from_parameter to create a benchmark ID with no function
        // name. That's intended for use with BenchmarkGroups where the function name isn't necessary,
        // but here it is.
        let group_name = id.function_name.expect(
            "Cannot use BenchmarkId::from_parameter with Criterion::bench_with_input. \
                 Consider using a BenchmarkGroup or BenchmarkId::new instead.",
        );
        // Guaranteed safe because external callers can't create benchmark IDs without a parameter
        let parameter = id.parameter.unwrap();
        self.benchmark_group(group_name).bench_with_input(
            BenchmarkId::no_function_with_input(parameter),
            input,
            f,
        );
        self
    }
}
