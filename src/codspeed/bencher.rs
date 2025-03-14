#[cfg(feature = "async")]
use std::future::Future;
use std::{cell::RefCell, rc::Rc};

use codspeed::codspeed::{CodSpeed, black_box};
use colored::Colorize;

#[cfg(feature = "async")]
use crate::async_executor::AsyncExecutor;
use crate::{BatchSize, Measurement, WallTime};

pub struct Bencher<'a, M: Measurement = WallTime> {
    codspeed: Rc<RefCell<CodSpeed>>,
    uri: String,
    _marker: std::marker::PhantomData<&'a M>,
}

impl<'a> Bencher<'a> {
    pub fn new(codspeed: Rc<RefCell<CodSpeed>>, uri: String) -> Self {
        Bencher { codspeed, uri, _marker: std::marker::PhantomData }
    }

    #[inline(never)]
    pub fn iter<O, R>(&mut self, mut routine: R)
    where
        R: FnMut() -> O,
    {
        let mut codspeed = self.codspeed.borrow_mut();
        // NOTE: this structure hardens our benchmark against dead code elimination
        // https://godbolt.org/z/KnYeKMd1o
        for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
            if i < codspeed::codspeed::WARMUP_RUNS {
                black_box(routine());
            } else {
                codspeed.start_benchmark(self.uri.as_str());
                black_box(routine());
                codspeed.end_benchmark();
            }
        }
    }

    #[inline(never)]
    pub fn iter_custom<R, MV>(&mut self, mut _routine: R)
    where
        R: FnMut(u64) -> MV,
    {
        println!(
            "{} {} (CodSpeed doesn't support custom iterations)",
            "Skipping:".to_string().yellow(),
            self.uri.yellow(),
        );
    }

    #[inline(never)]
    pub fn iter_batched<I, O, S, R>(&mut self, mut setup: S, mut routine: R, _size: BatchSize)
    where
        S: FnMut() -> I,
        R: FnMut(I) -> O,
    {
        let mut codspeed = self.codspeed.borrow_mut();

        for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
            let input = black_box(setup());
            let output = if i < codspeed::codspeed::WARMUP_RUNS {
                black_box(routine(input))
            } else {
                let input = black_box(setup());
                codspeed.start_benchmark(self.uri.as_str());
                let output = black_box(routine(input));
                codspeed.end_benchmark();
                output
            };
            drop(black_box(output));
        }
    }

    pub fn iter_with_setup<I, O, S, R>(&mut self, setup: S, routine: R)
    where
        S: FnMut() -> I,
        R: FnMut(I) -> O,
    {
        self.iter_batched(setup, routine, BatchSize::PerIteration);
    }

    pub fn iter_with_large_drop<O, R>(&mut self, mut routine: R)
    where
        R: FnMut() -> O,
    {
        self.iter_batched(|| (), |_| routine(), BatchSize::SmallInput);
    }

    pub fn iter_with_large_setup<I, O, S, R>(&mut self, setup: S, routine: R)
    where
        S: FnMut() -> I,
        R: FnMut(I) -> O,
    {
        self.iter_batched(setup, routine, BatchSize::NumBatches(1));
    }

    #[inline(never)]
    pub fn iter_batched_ref<I, O, S, R>(&mut self, mut setup: S, mut routine: R, _size: BatchSize)
    where
        S: FnMut() -> I,
        R: FnMut(&mut I) -> O,
    {
        let mut codspeed = self.codspeed.borrow_mut();

        for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
            let mut input = black_box(setup());
            let output = if i < codspeed::codspeed::WARMUP_RUNS {
                black_box(routine(&mut input))
            } else {
                codspeed.start_benchmark(self.uri.as_str());
                let output = black_box(routine(&mut input));
                codspeed.end_benchmark();
                output
            };
            drop(black_box(output));
            drop(black_box(input));
        }
    }

    #[inline(never)]
    pub fn iter_with_setup_wrapper<S>(&mut self, mut setup: S)
    where
        S: FnMut(&mut WrapperRunner),
    {
        let mut codspeed = self.codspeed.borrow_mut();
        let name = self.uri.as_str();

        for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
            if i < codspeed::codspeed::WARMUP_RUNS {
                WrapperRunner::execute(None, name, &mut setup)
            } else {
                WrapperRunner::execute(Some(&mut *codspeed), name, &mut setup)
            }
        }
    }

    #[cfg(feature = "async")]
    pub fn to_async<'b, A: AsyncExecutor>(&'b mut self, runner: A) -> AsyncBencher<'a, 'b, A> {
        AsyncBencher { b: self, runner }
    }
}

/// Runner used by [`Bencher::iter_with_setup_wrapper`].
pub struct WrapperRunner<'c> {
    codspeed: Option<&'c mut CodSpeed>,
    name: &'c str,
    has_run: bool,
}

impl<'c> WrapperRunner<'c> {
    fn execute<S>(codspeed: Option<&'c mut CodSpeed>, name: &'c str, setup: &mut S)
    where
        S: FnMut(&mut Self),
    {
        let mut runner = Self { codspeed, name, has_run: false };
        setup(&mut runner);
        assert!(runner.has_run, "setup function must call `WrapperRunner::run`");
    }

    pub fn run<O, R: FnOnce() -> O>(&mut self, routine: R) -> O {
        assert!(!self.has_run, "setup function must call `WrapperRunner::run` only once");
        self.has_run = true;

        let output = if let Some(codspeed) = self.codspeed.as_mut() {
            codspeed.start_benchmark(self.name);
            let output = black_box(routine());
            codspeed.end_benchmark();
            output
        } else {
            routine()
        };
        black_box(output)
    }
}

#[cfg(feature = "async")]
pub struct AsyncBencher<'a, 'b, A: AsyncExecutor> {
    b: &'b mut Bencher<'a>,
    runner: A,
}

#[cfg(feature = "async")]
impl<'a, 'b, A: AsyncExecutor> AsyncBencher<'a, 'b, A> {
    #[allow(clippy::await_holding_refcell_ref)]
    #[inline(never)]
    pub fn iter<O, R, F>(&mut self, mut routine: R)
    where
        R: FnMut() -> F,
        F: Future<Output = O>,
    {
        let AsyncBencher { b, runner } = self;
        runner.block_on(async {
            let mut codspeed = b.codspeed.borrow_mut();
            for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
                if i < codspeed::codspeed::WARMUP_RUNS {
                    black_box(routine().await);
                } else {
                    codspeed.start_benchmark(b.uri.as_str());
                    black_box(routine().await);
                    codspeed.end_benchmark();
                }
            }
        });
    }

    #[inline(never)]
    pub fn iter_custom<R, F, MV>(&mut self, mut _routine: R)
    where
        R: FnMut(u64) -> F,
        F: Future<Output = MV>,
    {
        let AsyncBencher { b, .. } = self;
        println!(
            "{} {} (CodSpeed doesn't support custom iterations)",
            "Skipping:".to_string().yellow(),
            b.uri.yellow(),
        );
    }

    #[doc(hidden)]
    pub fn iter_with_setup<I, O, S, R, F>(&mut self, setup: S, routine: R)
    where
        S: FnMut() -> I,
        R: FnMut(I) -> F,
        F: Future<Output = O>,
    {
        self.iter_batched(setup, routine, BatchSize::PerIteration);
    }

    pub fn iter_with_large_drop<O, R, F>(&mut self, mut routine: R)
    where
        R: FnMut() -> F,
        F: Future<Output = O>,
    {
        self.iter_batched(|| (), |_| routine(), BatchSize::SmallInput);
    }

    #[doc(hidden)]
    pub fn iter_with_large_setup<I, O, S, R, F>(&mut self, setup: S, routine: R)
    where
        S: FnMut() -> I,
        R: FnMut(I) -> F,
        F: Future<Output = O>,
    {
        self.iter_batched(setup, routine, BatchSize::NumBatches(1));
    }

    #[allow(clippy::await_holding_refcell_ref)]
    #[inline(never)]
    pub fn iter_batched<I, O, S, R, F>(&mut self, mut setup: S, mut routine: R, _size: BatchSize)
    where
        S: FnMut() -> I,
        R: FnMut(I) -> F,
        F: Future<Output = O>,
    {
        let AsyncBencher { b, runner } = self;
        runner.block_on(async {
            let mut codspeed = b.codspeed.borrow_mut();

            for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
                let input = black_box(setup());
                let output = if i < codspeed::codspeed::WARMUP_RUNS {
                    black_box(routine(input).await)
                } else {
                    codspeed.start_benchmark(b.uri.as_str());
                    let output = black_box(routine(input).await);
                    codspeed.end_benchmark();
                    output
                };
                drop(black_box(output));
            }
        })
    }

    #[allow(clippy::await_holding_refcell_ref)]
    #[inline(never)]
    pub fn iter_batched_ref<I, O, S, R, F>(
        &mut self,
        mut setup: S,
        mut routine: R,
        _size: BatchSize,
    ) where
        S: FnMut() -> I,
        R: FnMut(&mut I) -> F,
        F: Future<Output = O>,
    {
        let AsyncBencher { b, runner } = self;
        runner.block_on(async {
            let mut codspeed = b.codspeed.borrow_mut();

            for i in 0..codspeed::codspeed::WARMUP_RUNS + 1 {
                let mut input = black_box(setup());
                let output = if i < codspeed::codspeed::WARMUP_RUNS {
                    black_box(routine(&mut input).await)
                } else {
                    codspeed.start_benchmark(b.uri.as_str());
                    let output = black_box(routine(&mut input).await);
                    codspeed.end_benchmark();
                    output
                };
                drop(black_box(output));
                drop(black_box(input));
            }
        });
    }

    pub fn iter_with_setup_wrapper<S>(&mut self, mut setup: S)
    where
        S: FnMut(&mut WrapperRunner),
    {
        unimplemented!("Unsupported at present");
    }
}
