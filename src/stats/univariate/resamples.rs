use std::mem;

use crate::stats::{
    float::Float,
    rand_util::{Rng, new_rng},
    univariate::Sample,
};

pub struct Resamples<'a, A>
where
    A: Float,
{
    rng: Rng,
    sample: &'a [A],
    stage: Option<Vec<A>>,
}

impl<'a, A> Resamples<'a, A>
where
    A: 'a + Float,
{
    pub fn new(sample: &'a Sample<A>) -> Resamples<'a, A> {
        let slice = sample;

        Resamples { rng: new_rng(), sample: slice, stage: None }
    }

    pub fn next(&mut self) -> &Sample<A> {
        let n = self.sample.len();
        let rng = &mut self.rng;

        match self.stage {
            None => {
                let mut stage = Vec::with_capacity(n);

                for _ in 0..n {
                    let idx = rng.rand_range(0u64..(self.sample.len() as u64));
                    stage.push(self.sample[idx as usize])
                }

                self.stage = Some(stage);
            }
            Some(ref mut stage) => {
                for elem in stage.iter_mut() {
                    let idx = rng.rand_range(0u64..(self.sample.len() as u64));
                    *elem = self.sample[idx as usize]
                }
            }
        }

        if let Some(ref v) = self.stage {
            #[allow(clippy::missing_transmute_annotations)]
            unsafe {
                mem::transmute::<&[_], _>(v)
            }
        } else {
            unreachable!();
        }
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashSet;

    use quickcheck::{TestResult, quickcheck};

    use crate::stats::univariate::{Sample, resamples::Resamples};

    // Check that the resample is a subset of the sample
    quickcheck! {
        fn subset(size: u8, nresamples: u8) -> TestResult {
            let size = size as usize;
            let nresamples = nresamples as usize;
            if size > 1 {
                let v: Vec<_> = (0..size).map(|i| i as f32).collect();
                let sample = Sample::new(&v);
                let mut resamples = Resamples::new(sample);
                let sample = v.iter().map(|&x| x as i64).collect::<HashSet<_>>();

                TestResult::from_bool((0..nresamples).all(|_| {
                    let resample = resamples.next()

                        .iter()
                        .map(|&x| x as i64)
                        .collect::<HashSet<_>>();

                    resample.is_subset(&sample)
                }))
            } else {
                TestResult::discard()
            }
        }
    }

    #[test]
    fn different_subsets() {
        let size = 1000;
        let v: Vec<_> = (0..size).map(|i| i as f32).collect();
        let sample = Sample::new(&v);
        let mut resamples = Resamples::new(sample);

        // Hypothetically, we might see one duplicate, but more than one is likely to be a bug.
        let mut num_duplicated = 0;
        for _ in 0..1000 {
            let sample_1 = resamples.next().iter().cloned().collect::<Vec<_>>();
            let sample_2 = resamples.next().iter().cloned().collect::<Vec<_>>();

            if sample_1 == sample_2 {
                num_duplicated += 1;
            }
        }

        if num_duplicated > 1 {
            panic!("Found {} duplicate samples", num_duplicated);
        }
    }
}
