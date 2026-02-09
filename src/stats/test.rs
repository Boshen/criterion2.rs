use rand::{
    distr::{Distribution, StandardUniform},
    rngs::StdRng,
    RngExt, SeedableRng,
};

pub fn vec<T>(size: usize, start: usize) -> Option<Vec<T>>
where
    StandardUniform: Distribution<T>,
{
    if size > start + 2 {
        let seed = ((start as u64) << 32) ^ (size as u64);
        let mut rng = StdRng::seed_from_u64(seed);

        Some((0..size).map(|_| rng.random()).collect())
    } else {
        None
    }
}
