use rand::{
    distr::{Distribution, StandardUniform},
    prelude::*,
    rngs::StdRng,
};

pub fn vec<T>(size: usize, start: usize) -> Option<Vec<T>>
where
    StandardUniform: Distribution<T>,
{
    if size > start + 2 {
        let mut rng = StdRng::from_os_rng();

        Some((0..size).map(|_| rng.random()).collect())
    } else {
        None
    }
}
