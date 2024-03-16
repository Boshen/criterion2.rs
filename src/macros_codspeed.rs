#[macro_export]
macro_rules! abs_file {
    () => {
        std::path::PathBuf::from(
            std::env::var("CODSPEED_CARGO_WORKSPACE_ROOT")
            .expect("Could not find CODSPEED_CARGO_WORKSPACE_ROOT env variable, make sure you are using the latest version of cargo-codspeed")
        )
        .join(file!())
        .to_string_lossy()
    };
}

#[macro_export]
macro_rules! criterion_group {
    (name = $name:ident; config = $config:expr; targets = $( $target:path ),+ $(,)*) => {
        pub fn $name(criterion: &mut $crate::codspeed::criterion::Criterion) {
            let mut criterion = &mut criterion.with_patched_measurement($config);
            $(
                criterion.set_current_file(criterion::abs_file!());
                criterion.set_macro_group(format!("{}::{}", stringify!($name), stringify!($target)));
                $target(criterion);
            )+
        }
    };
    ($name:ident, $( $target:path ),+ $(,)*) => {
        $crate::criterion_group!{
            name = $name;
            config = $crate::Criterion::default();
            targets = $( $target ),+
        }
    }
}

#[macro_export]
macro_rules! criterion_main {
    ( $( $group:path ),+ $(,)* ) => {
        pub fn main() {
            let mut criterion = $crate::codspeed::criterion::Criterion::new_instrumented();
            $(
                $group(&mut criterion);
            )+
        }
    };
}
