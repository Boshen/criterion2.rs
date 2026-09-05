use std::{path::Path, str::FromStr, sync::OnceLock, time::Duration};

use usage::{ArgGroup, Cli};

use crate::{BenchmarkConfig, ListFormat, report::CliVerbosity};

#[derive(Debug, Clone)]
pub struct Opts {
    pub color: Color,
    pub verbosity: CliVerbosity,
    pub noplot: bool,
    pub filter: Option<String>,
    pub baseline: Baseline_,
    pub format: ListFormat,
    pub sample: Sample,
    pub op: Op,
    pub ignored: bool,
    pub exact: bool,
    pub warm_up_time: Duration,
    pub measurement_time: Duration,
    pub nresamples: usize,
    pub noise_threshold: f64,
    pub confidence_level: f64,
    pub significance_level: f64,
    pub output_format: OutputFormat,

    // ignored
    pub nocapture: bool,
    pub show_output: bool,
    pub include_ignored: bool,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OutputFormat {
    Criterion,
    Bencher,
}

impl FromStr for OutputFormat {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "criterion" => Self::Criterion,
            "bencher" => Self::Bencher,
            _ => return Err("Valid values are criterion and bencher"),
        })
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            OutputFormat::Criterion => "criterion",
            OutputFormat::Bencher => "bencher",
        })
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Color {
    Auto,
    Always,
    Never,
}

impl FromStr for Color {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "auto" => Self::Auto,
            "always" => Self::Always,
            "never" => Self::Never,
            _ => return Err("Valid values are auto, always and never"),
        })
    }
}

impl std::fmt::Display for Color {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Color::Auto => "auto",
            Color::Always => "always",
            Color::Never => "never",
        })
    }
}

#[derive(ArgGroup)]
#[usage(name = "verbosity")]
enum Verbosity {
    /// Print additional statistical information.
    #[usage(short = 'v', long = "verbose")]
    Verbose,

    /// Print only the benchmark results.
    Quiet,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Baseline_ {
    Save(String),
    Discard,
    Lenient(String),
    Strict(String),
}

#[derive(ArgGroup)]
#[usage(name = "baseline")]
enum Baseline {
    /// Save results under a named baseline.
    #[usage(short = 's', long = "save-baseline", value_name = "ARG")]
    Save(String),

    /// Discard benchmark results.
    #[usage(long = "discard-baseline")]
    Discard,

    /// Compare to a named baseline. If any benchmarks do not have the specified baseline this
    /// command fails.
    #[usage(short = 'b', long = "baseline", value_name = "BASE")]
    Strict(String),

    /// Compare to a named baseline. Benchmarks without the specified baseline are not compared,
    /// while every other benchmark is compared against the baseline.
    #[usage(long = "baseline-lenient", value_name = "BASE")]
    Lenient(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    List,
    LoadBaseline(String),
    ProfileTime(Duration),
    Test,
    Benchmark,
}

struct ProfileTime(Duration);

impl FromStr for ProfileTime {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let seconds = value.parse::<f64>().map_err(|error| error.to_string())?;
        if !(seconds > 1.0) {
            return Err("Profile time must be at least one second.".to_owned());
        }
        Ok(Self(Duration::from_secs_f64(seconds)))
    }
}

#[derive(ArgGroup)]
#[usage(name = "operation")]
enum Operation {
    /// List all benchmarks.
    List,

    #[usage(long = "profile-time", value_name = "DUR")]
    Profile(ProfileTime),

    /// Run the benchmarks once, to verify that they execute successfully, but do not measure or
    /// report the results.
    Test,

    /// Load a previous baseline instead of sampling new data.
    #[usage(long = "load-baseline", value_name = "BASE")]
    LoadBaseline(String),

    /// Run the benchmarks (default).
    #[usage(long = "bench")]
    Benchmark,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Sample {
    Specific(usize),
    Quick,
}

#[derive(ArgGroup)]
#[usage(name = "sample")]
enum Sampling {
    /// Benchmark only until the significance level has been reached.
    Quick,

    /// Changes the default size of the sample for this run.
    #[usage(long = "sample-size", value_name = "SIZE")]
    Specific(usize),
}

impl FromStr for ListFormat {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "pretty" => Self::Pretty,
            "terse" => Self::Terse,
            _ => return Err("Valid values are pretty and terse"),
        })
    }
}

const AFTER_HELP: &str = "This executable is a Criterion.rs benchmark.
See https://github.com/bheisler/criterion.rs for more details.

To enable debug output, define the environment variable CRITERION_DEBUG.
Criterion.rs will output more debug information and will save the gnuplot
scripts alongside the generated plots.

To test that the benchmarks work, run `cargo test --benches`

NOTE: If you see an 'unrecognized option' error using any of the options above, see:
https://bheisler.github.io/criterion.rs/book/faq.html";

#[derive(Cli)]
#[usage(
    name = executable_name(),
    name_spec = "benchmark",
    bin = executable_name(),
    bin_spec = "benchmark",
    unknown_flags = "error",
    args_override_self = false,
    spec_endpoint = false,
    after_help = AFTER_HELP
)]
struct RawOpts {
    /// Configure coloring of output. always = always colorize output, never = never colorize
    /// output, auto = colorize output if output is a tty and compiled for unix.
    #[usage(
        short = 'c',
        long,
        visible_alias = "colour",
        value_name = "COLOR",
        choices("auto", "always", "never"),
        default = "auto"
    )]
    color: Color,

    #[usage(arg_group)]
    verbosity: Option<Verbosity>,

    /// Disable plot and HTML generation.
    #[usage(short = 'n')]
    noplot: bool,

    #[usage(arg_group)]
    baseline: Option<Baseline>,

    #[usage(arg_group)]
    sample: Option<Sampling>,

    #[usage(arg_group)]
    op: Option<Operation>,

    /// Output formatting.
    #[usage(long, value_name = "FORMAT", choices("pretty", "terse"), default = "pretty")]
    format: ListFormat,

    /// Changes the default warm up time for this run.
    #[usage(long = "warm-up-time", value_name = "TIME")]
    warm_up_time: Option<f64>,

    /// Changes the default measurement time for this run.
    #[usage(long = "measurement-time", value_name = "TIME")]
    measurement_time: Option<f64>,

    /// Changes the default number of resamples for this run.
    #[usage(long, value_name = "N")]
    nresamples: Option<usize>,

    /// Changes the default noise threshold for this run.
    #[usage(long = "noise-threshold", value_name = "ARG")]
    noise_threshold: Option<f64>,

    /// Changes the default confidence level for this run.
    #[usage(long = "confidence-level", value_name = "ARG")]
    confidence_level: Option<f64>,

    /// Changes the default significance level for this run.
    #[usage(long = "significance-level", value_name = "ARG")]
    significance_level: Option<f64>,

    /// Ignored, but added for compatibility with libtest.
    #[usage(long, hide)]
    nocapture: bool,

    /// Ignored, but added for compatibility with libtest.
    #[usage(long = "show-output", hide)]
    show_output: bool,

    /// Ignored, but added for compatibility with libtest.
    #[usage(long = "include-ignored", hide)]
    include_ignored: bool,

    /// Change the CLI output format. By default, Criterion.rs will use its own format. If output
    /// format is set to 'bencher', Criterion.rs will print output in a format that resembles the
    /// 'bencher' crate.
    #[usage(
        long = "output-format",
        value_name = "FORMAT",
        choices("criterion", "bencher"),
        default = "criterion"
    )]
    output_format: OutputFormat,

    /// List or run ignored benchmarks (currently means skip all benchmarks).
    #[usage(long)]
    ignored: bool,

    /// Run benchmarks that exactly match the provided filter.
    #[usage(long)]
    exact: bool,

    /// Skip benchmarks whose names do not contain FILTER.
    filter: Option<String>,
}

impl RawOpts {
    fn with_config(self, config: &BenchmarkConfig) -> Opts {
        let verbosity = match self.verbosity {
            Some(Verbosity::Verbose) => CliVerbosity::Verbose,
            Some(Verbosity::Quiet) => CliVerbosity::Quiet,
            None => CliVerbosity::Normal,
        };
        let baseline = match self.baseline {
            Some(Baseline::Save(name)) => Baseline_::Save(name),
            Some(Baseline::Discard) => Baseline_::Discard,
            Some(Baseline::Strict(name)) => Baseline_::Strict(name),
            Some(Baseline::Lenient(name)) => Baseline_::Lenient(name),
            None => Baseline_::Save("baseline".to_owned()),
        };
        let sample = match self.sample {
            Some(Sampling::Quick) => Sample::Quick,
            Some(Sampling::Specific(size)) => Sample::Specific(size),
            None => Sample::Specific(config.sample_size),
        };
        let op = match self.op {
            Some(Operation::List) => Op::List,
            Some(Operation::Profile(ProfileTime(time))) => Op::ProfileTime(time),
            Some(Operation::Test) => Op::Test,
            Some(Operation::LoadBaseline(name)) => Op::LoadBaseline(name),
            Some(Operation::Benchmark) | None => Op::Benchmark,
        };

        Opts {
            color: self.color,
            verbosity,
            noplot: self.noplot,
            filter: self.filter,
            baseline,
            format: self.format,
            sample,
            op,
            ignored: self.ignored,
            exact: self.exact,
            warm_up_time: self.warm_up_time.map_or(config.warm_up_time, Duration::from_secs_f64),
            measurement_time: self
                .measurement_time
                .map_or(config.measurement_time, Duration::from_secs_f64),
            nresamples: self.nresamples.unwrap_or(config.nresamples),
            noise_threshold: self.noise_threshold.unwrap_or(config.noise_threshold),
            confidence_level: self.confidence_level.unwrap_or(config.confidence_level),
            significance_level: self.significance_level.unwrap_or(config.significance_level),
            output_format: self.output_format,
            nocapture: self.nocapture,
            show_output: self.show_output,
            include_ignored: self.include_ignored,
        }
    }
}

fn executable_name() -> &'static str {
    static NAME: OnceLock<String> = OnceLock::new();
    NAME.get_or_init(|| {
        std::env::args_os()
            .next()
            .and_then(|argument| Path::new(&argument).file_name().map(|name| name.to_owned()))
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "benchmark".to_owned())
    })
}

pub fn parse(config: &BenchmarkConfig) -> Opts {
    RawOpts::parse().with_config(config)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::*;

    fn parse(args: &[&str], config: &BenchmarkConfig) -> Opts {
        let args: Vec<&OsStr> = args.iter().map(OsStr::new).collect();
        RawOpts::parse_from(&args).unwrap().with_config(config)
    }

    #[test]
    fn uses_benchmark_configuration_as_defaults() {
        let mut config = BenchmarkConfig::default();
        config.sample_size = 42;
        config.nresamples = 1234;

        let opts = parse(&[], &config);

        assert_eq!(opts.sample, Sample::Specific(42));
        assert_eq!(opts.nresamples, 1234);
        assert_eq!(opts.op, Op::Benchmark);
        assert_eq!(opts.baseline, Baseline_::Save("baseline".to_owned()));
    }

    #[test]
    fn parses_alternative_modes_and_aliases() {
        let config = BenchmarkConfig::default();
        let opts = parse(
            &[
                "--colour",
                "always",
                "--quiet",
                "--baseline-lenient",
                "previous",
                "--quick",
                "--list",
                "--format",
                "terse",
                "--exact",
                "filter",
            ],
            &config,
        );

        assert_eq!(opts.color, Color::Always);
        assert_eq!(opts.verbosity, CliVerbosity::Quiet);
        assert_eq!(opts.baseline, Baseline_::Lenient("previous".to_owned()));
        assert_eq!(opts.sample, Sample::Quick);
        assert_eq!(opts.op, Op::List);
        assert!(matches!(opts.format, ListFormat::Terse));
        assert!(opts.exact);
        assert_eq!(opts.filter.as_deref(), Some("filter"));
    }

    #[test]
    fn rejects_conflicting_modes() {
        let args = [OsStr::new("--quick"), OsStr::new("--sample-size"), OsStr::new("20")];

        assert!(RawOpts::parse_from(&args).is_err());
    }

    #[test]
    fn rejects_short_profile_times() {
        let args = [OsStr::new("--profile-time"), OsStr::new("1")];

        assert!(RawOpts::parse_from(&args).is_err());
    }
}
