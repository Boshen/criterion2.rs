use criterion::{criterion_group, criterion_main, Criterion};

#[test]
#[should_panic(expected = "group executed")]
fn criterion_main() {
    fn group() {}
    fn group2() {
        panic!("group executed");
    }

    criterion_main!(group, group2);

    main();
}

#[test]
fn criterion_main_trailing_comma() {
    // make this a compile-only check
    // as the second logger initialization causes panic
    #[allow(dead_code)]
    fn group() {}
    #[allow(dead_code)]
    fn group2() {}

    criterion_main!(group, group2,);

    // silence dead_code warning
    if false {
        main()
    }
}

#[test]
#[should_panic(expected = "group executed")]
fn criterion_group() {
    fn group(_crit: &mut Criterion) {}
    fn group2(_crit: &mut Criterion) {
        panic!("group executed");
    }

    criterion_group!(test_group, group, group2);

    test_group();
}

#[test]
#[should_panic(expected = "group executed")]
fn criterion_group_trailing_comma() {
    fn group(_crit: &mut Criterion) {}
    fn group2(_crit: &mut Criterion) {
        panic!("group executed");
    }

    criterion_group!(test_group, group, group2,);

    test_group();
}
