use crate::{report::CliVerbosity, BenchmarkConfig, ListFormat, PlottingBackend};
use bpaf::*;
use std::{str::FromStr, time::Duration};

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
    pub plotting_backend: Option<PlottingBackend>,
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

fn verbosity() -> impl Parser<CliVerbosity> {
    let verbose = short('v')
        .long("verbose")
        .help("Print additional statistical information.")
        .req_flag(CliVerbosity::Verbose);

    let quiet =
        long("quiet").help("Print only the benchmark results.").req_flag(CliVerbosity::Quiet);
    construct!([verbose, quiet]).fallback(CliVerbosity::Normal)
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Baseline_ {
    Save(String),
    Discard,
    Lenient(String),
    Strict(String),
}

fn baseline() -> impl Parser<Baseline_> {
    let save_baseline = short('s')
        .long("save-baseline")
        .help("Save results under a named baseline.")
        .argument("ARG")
        .map(Baseline_::Save);

    let discard_baseline =
        long("discard-baseline").help("Discard benchmark results").req_flag(Baseline_::Discard);

    let use_baseline = short('b')
        .long("baseline")
        .help(
            "Compare to a named baseline. If any benchmarks do not have the specified baseline \
                            this command fails.",
        )
        .argument::<String>("BASE")
        .map(Baseline_::Strict);

    let baseline_lenient =
                        long("baseline-lenient")

                        .help("Compare to a named baseline. If any benchmarks do not have the specified baseline \
                            then just those benchmarks are not compared against the baseline while every other \
                            benchmark is compared against the baseline.")
                        .argument("BASE")
                        .map(Baseline_::Lenient);

    construct!([save_baseline, discard_baseline, use_baseline, baseline_lenient])
        .fallback(Baseline_::Save("baseline".to_owned()))
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    List,
    LoadBaseline(String),
    ProfileTime(Duration),
    Test,
    Benchmark,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Sample {
    Specific(usize),
    Quick,
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
fn format() -> impl Parser<ListFormat> {
    // Note that libtest's --format also works during test execution, but criterion
    // doesn't support that at the moment.
    long("format").help("Output formatting").argument("FORMAT").fallback(ListFormat::Pretty)
}

fn sample(default_sample: usize) -> impl Parser<Sample> {
    let quick = long("quick")
        .help("Benchmark only until the significance level has been reached")
        .req_flag(Sample::Quick);
    let sample = long("sample-size")
        .help("Changes the default size of the sample for this run")
        .argument::<usize>("SIZE")
        .fallback(default_sample)
        .display_fallback()
        .map(Sample::Specific);
    construct!([quick, sample])
}

fn op() -> impl Parser<Op> {
    let list = long("list").help("List all benchmarks").req_flag(Op::List);
    let profile = long("profile-time")
        .argument::<f64>("DUR")
        .guard(|d| *d > 1.0, "Profile time must be at least one second.")
        .map(|s| Op::ProfileTime(Duration::from_secs_f64(s)));
    let test = long("test")
        .help(
            "Run the benchmarks once, to verify that they execute successfully, \
                            but do not measure or report the results.",
        )
        .req_flag(Op::Test);
    let load_baseline = long("load-baseline")
        .help("Load a previous baseline instead of sampling new data.")
        .argument::<String>("BASE")
        .map(Op::LoadBaseline);
    let bench = long("bench").help("Run the benchmarks (default)").req_flag(Op::Benchmark);
    construct!([list, profile, load_baseline, test, bench])
        .fallback(Op::Benchmark)
        .group_help("Operation to perform")
}

pub fn options(config: &BenchmarkConfig) -> OptionParser<Opts> {
    let filter = positional::<String>("FILTER")
        .help("Skip benchmarks whose names do not contain FILTER.")
        .optional();

    let color = short('c')
        .long("color")
        .long("colour")
        .help(
            "Configure coloring of output. always = always colorize output, \
                    never = never colorize output, auto = colorize output if output \
                    is a tty and compiled for unix.",
        )
        .argument::<Color>("COLOR")
        .fallback(Color::Auto);

    let noplot = short('n').help("Disable plot and HTML generation.").switch();

    let ignored = long("ignored")
        .help("List or run ignored benchmarks (currently means skip all benchmarks)")
        .switch();

    let exact =
        long("exact").help("Run benchmarks that exactly match the provided filter").switch();

    let sample = sample(config.sample_size);

    let warm_up_time = long("warm-up-time")
        .help("Changes the default warm up time for this run")
        .argument("TIME")
        .fallback(config.warm_up_time.as_secs_f64())
        .display_fallback()
        .map(Duration::from_secs_f64);

    let measurement_time = long("measurement-time")
        .help("Changes the default measurement time for this run.")
        .argument::<f64>("TIME")
        .fallback(config.measurement_time.as_secs_f64())
        .display_fallback()
        .map(Duration::from_secs_f64);

    let nresamples = long("nresamples")
        .help("Changes the default number of resamples for this run")
        .argument::<usize>("N")
        .fallback(config.nresamples)
        .display_fallback();
    let noise_threshold = long("noise-threshold")
        .help("Changes the default noise threshold for this run.")
        .argument::<f64>("ARG")
        .fallback(config.noise_threshold)
        .display_fallback();
    let confidence_level = long("confidence-level")
        .help("Changes the default confidence level for this run.")
        .argument::<f64>("ARG")
        .fallback(config.confidence_level)
        .display_fallback();
    let significance_level = long("significance-level")
        .help("Changes the default significance level for this run.")
        .argument::<f64>("ARG")
        .fallback(config.significance_level)
        .display_fallback();
    let nocapture = long("nocapture")
        .help("Ignored, but added for compatibility with libsets.")
        .switch()
        .hide();
    let show_output = long("show-output")
        .help("Ignored, but added for compatibility with libsets.")
        .switch()
        .hide();
    let include_ignored = long("include-ignored")
        .help("Ignored, but added for compatibility with libsets.")
        .switch()
        .hide();
    let plotting_backend = long("plotting-backend")
        .help(
            "Set the plotting backend. By default, Criterion.rs will use the gnuplot backend \
                             if gnuplot is available, or the plotters backend if it isn't.",
        )
        .argument("PLOT")
        .optional();

    let output_format =
                        long("output-format")

                        .help("Change the CLI output format. By default, Criterion.rs will use its own \
                             format. If output format is set to 'bencher', Criterion.rs will print output \
                             in a format that resembles the 'bencher' crate.")
                        .argument("FORMAT").fallback(OutputFormat::Criterion);

    construct!(Opts { color, verbosity(),  noplot, baseline(), sample, op(), format(),
        warm_up_time, measurement_time,
        nresamples, noise_threshold, confidence_level, significance_level,
        nocapture, show_output, include_ignored,
        plotting_backend,
        output_format,
        ignored, exact, filter})
    .to_options()
    .footer(
        "
This executable is a Criterion.rs benchmark.
 See https://github.com/bheisler/criterion.rs for more details.

  To enable debug output, define the environment variable CRITERION_DEBUG.
  Criterion.rs will output more debug information and will save the gnuplot
  scripts alongside the generated plots.

  To test that the benchmarks work, run `cargo test --benches`

  NOTE: If you see an 'unrecognized option' error using any of the options above, see:
  https://bheisler.github.io/criterion.rs/book/faq.html",
    )
}

#[test]
fn check_invariants() {
    let cfg = BenchmarkConfig::default();
    let parser = options(&cfg);

    parser.check_invariants(true);
}
