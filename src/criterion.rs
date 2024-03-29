use std::cell::RefCell;
use std::collections::HashSet;
use std::io::{stdout, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::MutexGuard;
use std::time::Duration;

use regex::Regex;

use crate::bencher::Bencher;
use crate::benchmark_group::{BenchmarkGroup, BenchmarkId};
use crate::{
    cargo_criterion_connection, debug_enabled, default_output_directory, default_plotting_backend,
    gnuplot_version, Baseline, BencherReport, BenchmarkConfig, BenchmarkFilter, CliReport,
    CliVerbosity, Connection, ExternalProfiler, Html, ListFormat, Measurement, Mode,
    OutgoingMessage, PlotConfiguration, PlottingBackend, Profiler, Report, ReportContext, Reports,
    SamplingMode, WallTime,
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
            html: default_plotting_backend().create_plotter().map(Html::new),
            csv_enabled: cfg!(feature = "csv_output"),
        };

        let mut criterion = Criterion {
            config: BenchmarkConfig {
                confidence_level: 0.95,
                measurement_time: Duration::from_secs(5),
                noise_threshold: 0.01,
                nresamples: 100_000,
                sample_size: 100,
                significance_level: 0.05,
                warm_up_time: Duration::from_secs(3),
                sampling_mode: SamplingMode::Auto,
                quick_mode: false,
            },
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
            criterion.report.csv_enabled = false;
            criterion.report.html = None;
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
    /// Set the plotting backend. By default, Criterion will use gnuplot if available, or plotters
    /// if not.
    ///
    /// Panics if `backend` is `PlottingBackend::Gnuplot` and gnuplot is not available.
    pub fn plotting_backend(mut self, backend: PlottingBackend) -> Criterion<M> {
        if let PlottingBackend::Gnuplot = backend {
            assert!(
                !gnuplot_version().is_err(),
                "Gnuplot plotting backend was requested, but gnuplot is not available. \
                To continue, either install Gnuplot or allow Criterion.rs to fall back \
                to using plotters."
            );
        }

        self.report.html = backend.create_plotter().map(Html::new);
        self
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
    /// Enables plotting
    pub fn with_plots(mut self) -> Criterion<M> {
        // If running under cargo-criterion then don't re-enable the reports; let it do the reporting.
        if self.connection.is_none() && self.report.html.is_none() {
            let default_backend = default_plotting_backend().create_plotter();
            if let Some(backend) = default_backend {
                self.report.html = Some(Html::new(backend));
            } else {
                panic!("Cannot find a default plotting backend!");
            }
        }
        self
    }

    #[must_use]
    /// Disables plotting
    pub fn without_plots(mut self) -> Criterion<M> {
        self.report.html = None;
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

    #[must_use]
    /// Filters the benchmarks. Only benchmarks with names that contain the
    /// given string will be executed.
    ///
    /// This overwrites [`Self::with_benchmark_filter`].
    pub fn with_filter<S: Into<String>>(mut self, filter: S) -> Criterion<M> {
        let filter_text = filter.into();
        let filter = Regex::new(&filter_text).unwrap_or_else(|err| {
            panic!("Unable to parse '{}' as a regular expression: {}", filter_text, err)
        });
        self.filter = BenchmarkFilter::Regex(filter);

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
        self.output_directory = path.to_owned();

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

        let report_context = ReportContext {
            output_directory: self.output_directory.clone(),
            plot_config: PlotConfiguration::default(),
        };

        self.report.final_summary(&report_context);
    }

    /// Configure this criterion struct based on the command-line arguments to
    /// this process.
    #[must_use]
    pub fn configure_from_args(mut self) -> Criterion<M> {
        use clap::{value_parser, Arg, Command};
        let matches = Command::new("Criterion Benchmark")
            .arg(Arg::new("FILTER")
                .help("Skip benchmarks whose names do not contain FILTER.")
                .index(1))
            .arg(Arg::new("color")
                .short('c')
                .long("color")
                .alias("colour")
                .value_parser(["auto", "always", "never"])
                .default_value("auto")
                .help("Configure coloring of output. always = always colorize output, never = never colorize output, auto = colorize output if output is a tty and compiled for unix."))
            .arg(Arg::new("verbose")
                .short('v')
                .long("verbose")
                .num_args(0)
                .help("Print additional statistical information."))
            .arg(Arg::new("quiet")
                .long("quiet")
                .num_args(0)
                .conflicts_with("verbose")
                .help("Print only the benchmark results."))
            .arg(Arg::new("noplot")
                .short('n')
                .long("noplot")
                .num_args(0)
                .help("Disable plot and HTML generation."))
            .arg(Arg::new("save-baseline")
                .short('s')
                .long("save-baseline")
                .default_value("base")
                .help("Save results under a named baseline."))
            .arg(Arg::new("discard-baseline")
                .long("discard-baseline")
                .num_args(0)
                .conflicts_with_all(["save-baseline", "baseline", "baseline-lenient"])
                .help("Discard benchmark results."))
            .arg(Arg::new("baseline")
                .short('b')
                .long("baseline")
                .conflicts_with_all(["save-baseline", "baseline-lenient"])
                .help("Compare to a named baseline. If any benchmarks do not have the specified baseline this command fails."))
            .arg(Arg::new("baseline-lenient")
                .long("baseline-lenient")
                .conflicts_with_all(["save-baseline", "baseline"])
                .help("Compare to a named baseline. If any benchmarks do not have the specified baseline then just those benchmarks are not compared against the baseline while every other benchmark is compared against the baseline."))
            .arg(Arg::new("list")
                .long("list")
                .num_args(0)
                .help("List all benchmarks")
                .conflicts_with_all(["test", "profile-time"]))
            .arg(Arg::new("format")
                .long("format")
                .value_parser(["pretty", "terse"])
                .default_value("pretty")
                // Note that libtest's --format also works during test execution, but criterion
                // doesn't support that at the moment.
                .help("Output formatting"))
            .arg(Arg::new("ignored")
                .long("ignored")
                .num_args(0)
                .help("List or run ignored benchmarks (currently means skip all benchmarks)"))
            .arg(Arg::new("exact")
                .long("exact")
                .num_args(0)
                .help("Run benchmarks that exactly match the provided filter"))
            .arg(Arg::new("profile-time")
                .long("profile-time")
                .value_parser(value_parser!(f64))
                .help("Iterate each benchmark for approximately the given number of seconds, doing no analysis and without storing the results. Useful for running the benchmarks in a profiler.")
                .conflicts_with_all(["test", "list"]))
            .arg(Arg::new("load-baseline")
                 .long("load-baseline")
                 .conflicts_with("profile-time")
                 .requires("baseline")
                 .help("Load a previous baseline instead of sampling new data."))
            .arg(Arg::new("sample-size")
                .long("sample-size")
                .value_parser(value_parser!(usize))
                .help(format!("Changes the default size of the sample for this run. [default: {}]", self.config.sample_size)))
            .arg(Arg::new("warm-up-time")
                .long("warm-up-time")
                .value_parser(value_parser!(f64))
                .help(format!("Changes the default warm up time for this run. [default: {}]", self.config.warm_up_time.as_secs())))
            .arg(Arg::new("measurement-time")
                .long("measurement-time")
                .value_parser(value_parser!(f64))
                .help(format!("Changes the default measurement time for this run. [default: {}]", self.config.measurement_time.as_secs())))
            .arg(Arg::new("nresamples")
                .long("nresamples")
                .value_parser(value_parser!(usize))
                .help(format!("Changes the default number of resamples for this run. [default: {}]", self.config.nresamples)))
            .arg(Arg::new("noise-threshold")
                .long("noise-threshold")
                .value_parser(value_parser!(f64))
                .help(format!("Changes the default noise threshold for this run. [default: {}]", self.config.noise_threshold)))
            .arg(Arg::new("confidence-level")
                .long("confidence-level")
                .value_parser(value_parser!(f64))
                .help(format!("Changes the default confidence level for this run. [default: {}]", self.config.confidence_level)))
            .arg(Arg::new("significance-level")
                .long("significance-level")
                .value_parser(value_parser!(f64))
                .help(format!("Changes the default significance level for this run. [default: {}]", self.config.significance_level)))
            .arg(Arg::new("quick")
                .long("quick")
                .num_args(0)
                .conflicts_with("sample-size")
                .help(format!("Benchmark only until the significance level has been reached [default: {}]", self.config.quick_mode)))
            .arg(Arg::new("test")
                .hide(true)
                .long("test")
                .num_args(0)
                .help("Run the benchmarks once, to verify that they execute successfully, but do not measure or report the results.")
                .conflicts_with_all(["list", "profile-time"]))
            .arg(Arg::new("bench")
                .hide(true)
                .long("bench")
                .num_args(0))
            .arg(Arg::new("plotting-backend")
                 .long("plotting-backend")
                 .value_parser(["gnuplot", "plotters"])
                 .help("Set the plotting backend. By default, Criterion.rs will use the gnuplot backend if gnuplot is available, or the plotters backend if it isn't."))
            .arg(Arg::new("output-format")
                .long("output-format")
                .value_parser(["criterion", "bencher"])
                .default_value("criterion")
                .help("Change the CLI output format. By default, Criterion.rs will use its own format. If output format is set to 'bencher', Criterion.rs will print output in a format that resembles the 'bencher' crate."))
            .arg(Arg::new("nocapture")
                .long("nocapture")
                .num_args(0)
                .hide(true)
                .help("Ignored, but added for compatibility with libtest."))
            .arg(Arg::new("show-output")
                .long("show-output")
                .num_args(0)
                .hide(true)
                .help("Ignored, but added for compatibility with libtest."))
            .arg(Arg::new("include-ignored")
                .long("include-ignored")
                .num_args(0)
                .hide(true)
                .help("Ignored, but added for compatibility with libtest."))
            .arg(Arg::new("version")
                .hide(true)
                .short('V')
                .long("version")
                .num_args(0))
            .after_help("
This executable is a Criterion.rs benchmark.
See https://github.com/bheisler/criterion.rs for more details.

To enable debug output, define the environment variable CRITERION_DEBUG.
Criterion.rs will output more debug information and will save the gnuplot
scripts alongside the generated plots.

To test that the benchmarks work, run `cargo test --benches`

NOTE: If you see an 'unrecognized option' error using any of the options above, see:
https://bheisler.github.io/criterion.rs/book/faq.html
")
            .get_matches();

        if self.connection.is_some() {
            if let Some(color) = matches.get_one::<String>("color") {
                if color != "auto" {
                    eprintln!("Warning: --color will be ignored when running with cargo-criterion. Use `cargo criterion --color {} -- <args>` instead.", color);
                }
            }
            if matches.get_flag("verbose") {
                eprintln!("Warning: --verbose will be ignored when running with cargo-criterion. Use `cargo criterion --output-format verbose -- <args>` instead.");
            }
            if matches.get_flag("noplot") {
                eprintln!("Warning: --noplot will be ignored when running with cargo-criterion. Use `cargo criterion --plotting-backend disabled -- <args>` instead.");
            }
            if let Some(backend) = matches.get_one::<String>("plotting-backend") {
                eprintln!("Warning: --plotting-backend will be ignored when running with cargo-criterion. Use `cargo criterion --plotting-backend {} -- <args>` instead.", backend);
            }
            if let Some(format) = matches.get_one::<String>("output-format") {
                if format != "criterion" {
                    eprintln!("Warning: --output-format will be ignored when running with cargo-criterion. Use `cargo criterion --output-format {} -- <args>` instead.", format);
                }
            }

            if matches.contains_id("baseline")
                || matches.get_one::<String>("save-baseline").map_or(false, |base| base != "base")
                || matches.contains_id("load-baseline")
            {
                eprintln!("Error: baselines are not supported when running with cargo-criterion.");
                std::process::exit(1);
            }
        }

        let bench = matches.get_flag("bench");
        let test = matches.get_flag("test");
        let test_mode = match (bench, test) {
            (true, true) => true,   // cargo bench -- --test should run tests
            (true, false) => false, // cargo bench should run benchmarks
            (false, _) => true,     // cargo test --benches should run tests
        };

        self.mode = if matches.get_flag("list") {
            let list_format = match matches
                .get_one::<String>("format")
                .expect("a default value was provided for this")
                .as_str()
            {
                "pretty" => ListFormat::Pretty,
                "terse" => ListFormat::Terse,
                other => unreachable!(
                    "unrecognized value for --format that isn't part of possible-values: {}",
                    other
                ),
            };
            Mode::List(list_format)
        } else if test_mode {
            Mode::Test
        } else if let Some(&num_seconds) = matches.get_one("profile-time") {
            if num_seconds < 1.0 {
                eprintln!("Profile time must be at least one second.");
                std::process::exit(1);
            }

            Mode::Profile(Duration::from_secs_f64(num_seconds))
        } else {
            Mode::Benchmark
        };

        // This is kind of a hack, but disable the connection to the runner if we're not benchmarking.
        if !self.mode.is_benchmark() {
            self.connection = None;
        }

        let filter = if matches.get_flag("ignored") {
            // --ignored overwrites any name-based filters passed in.
            BenchmarkFilter::RejectAll
        } else if let Some(filter) = matches.get_one::<String>("FILTER") {
            if matches.get_flag("exact") {
                BenchmarkFilter::Exact(filter.to_owned())
            } else {
                let regex = Regex::new(filter).unwrap_or_else(|err| {
                    panic!("Unable to parse '{}' as a regular expression: {}", filter, err)
                });
                BenchmarkFilter::Regex(regex)
            }
        } else {
            BenchmarkFilter::AcceptAll
        };
        self = self.with_benchmark_filter(filter);

        match matches.get_one("plotting-backend").map(String::as_str) {
            // Use plotting_backend() here to re-use the panic behavior if Gnuplot is not available.
            Some("gnuplot") => self = self.plotting_backend(PlottingBackend::Gnuplot),
            Some("plotters") => self = self.plotting_backend(PlottingBackend::Plotters),
            Some(val) => panic!("Unexpected plotting backend '{}'", val),
            None => {}
        }

        if matches.get_flag("noplot") {
            self = self.without_plots();
        }

        if let Some(dir) = matches.get_one::<String>("save-baseline") {
            self.baseline = Baseline::Save;
            self.baseline_directory = dir.to_owned()
        }
        if matches.get_flag("discard-baseline") {
            self.baseline = Baseline::Discard;
        }
        if let Some(dir) = matches.get_one::<String>("baseline") {
            self.baseline = Baseline::CompareStrict;
            self.baseline_directory = dir.to_owned();
        }
        if let Some(dir) = matches.get_one::<String>("baseline-lenient") {
            self.baseline = Baseline::CompareLenient;
            self.baseline_directory = dir.to_owned();
        }

        if self.connection.is_some() {
            // disable all reports when connected to cargo-criterion; it will do the reporting.
            self.report.cli_enabled = false;
            self.report.bencher_enabled = false;
            self.report.csv_enabled = false;
            self.report.html = None;
        } else {
            match matches.get_one("output-format").map(String::as_str) {
                Some("bencher") => {
                    self.report.bencher_enabled = true;
                    self.report.cli_enabled = false;
                }
                _ => {
                    let verbose = matches.get_flag("verbose");
                    let verbosity = if verbose {
                        CliVerbosity::Verbose
                    } else if matches.get_flag("quiet") {
                        CliVerbosity::Quiet
                    } else {
                        CliVerbosity::Normal
                    };
                    let stdout_isatty = stdout().is_terminal();
                    let mut enable_text_overwrite = stdout_isatty && !verbose && !debug_enabled();
                    let enable_text_coloring;
                    match matches.get_one("color").map(String::as_str) {
                        Some("always") => {
                            enable_text_coloring = true;
                        }
                        Some("never") => {
                            enable_text_coloring = false;
                            enable_text_overwrite = false;
                        }
                        _ => enable_text_coloring = stdout_isatty,
                    };
                    self.report.bencher_enabled = false;
                    self.report.cli_enabled = true;
                    self.report.cli =
                        CliReport::new(enable_text_overwrite, enable_text_coloring, verbosity);
                }
            };
        }

        if let Some(dir) = matches.get_one::<String>("load-baseline") {
            self.load_baseline = Some(dir.to_owned());
        }

        if let Some(&num_size) = matches.get_one("sample-size") {
            assert!(num_size >= 10);
            self.config.sample_size = num_size;
        }
        if let Some(&num_seconds) = matches.get_one("warm-up-time") {
            let dur = std::time::Duration::from_secs_f64(num_seconds);
            assert!(dur.as_nanos() > 0);

            self.config.warm_up_time = dur;
        }
        if let Some(&num_seconds) = matches.get_one("measurement-time") {
            let dur = std::time::Duration::from_secs_f64(num_seconds);
            assert!(dur.as_nanos() > 0);

            self.config.measurement_time = dur;
        }
        if let Some(&num_resamples) = matches.get_one("nresamples") {
            assert!(num_resamples > 0);

            self.config.nresamples = num_resamples;
        }
        if let Some(&num_noise_threshold) = matches.get_one("noise-threshold") {
            assert!(num_noise_threshold > 0.0);

            self.config.noise_threshold = num_noise_threshold;
        }
        if let Some(&num_confidence_level) = matches.get_one("confidence-level") {
            assert!(num_confidence_level > 0.0 && num_confidence_level < 1.0);

            self.config.confidence_level = num_confidence_level;
        }
        if let Some(&num_significance_level) = matches.get_one("significance-level") {
            assert!(num_significance_level > 0.0 && num_significance_level < 1.0);

            self.config.significance_level = num_significance_level;
        }

        if matches.get_flag("quick") {
            self.config.quick_mode = true;
        }

        self
    }

    pub(crate) fn filter_matches(&self, id: &str) -> bool {
        match &self.filter {
            BenchmarkFilter::AcceptAll => true,
            BenchmarkFilter::Regex(regex) => regex.is_match(id),
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
    /// use self::criterion2::*;
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
    /// use self::criterion2::*;
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
    /// use self::criterion2::*;
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
