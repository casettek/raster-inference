use crate::dsl::prelude::*;

#[derive(Debug, Clone, Copy)]
struct TestAuthSource;

#[derive(Debug, Clone, Copy)]
struct TestAuthRequest {
    fail: bool,
}

impl AuthRead<TestAuthRequest> for TestAuthSource {
    type Output = u32;

    fn auth_read(&self, request: TestAuthRequest) -> anyhow::Result<Self::Output> {
        if request.fail {
            anyhow::bail!("auth read failed");
        }
        Ok(42)
    }
}

#[tile]
fn add_one(value: u32) -> u32 {
    value + 1
}

#[sequence]
fn add_sequence(value: u32) -> u32 {
    call_tile!(add_one, value)
}

#[tile(kind = recur)]
fn fallible_double_until_at_least_ten(value: u32) -> anyhow::Result<(bool, u32)> {
    if value >= 10 {
        Ok((true, value))
    } else {
        Ok((false, value * 2))
    }
}

#[tile(kind = recur)]
fn fallible_count_until_context(current: u32, goal: u32) -> anyhow::Result<(bool, u32)> {
    if current >= goal {
        Ok((true, current))
    } else {
        Ok((false, current + 1))
    }
}

#[tile(kind = recur)]
fn fallible_step_that_fails(value: u32) -> anyhow::Result<(bool, u32)> {
    if value >= 2 {
        anyhow::bail!("step failed");
    }
    Ok((false, value + 1))
}

#[sequence(kind = recur)]
fn fallible_add_sequence_until_at_least_ten(value: u32) -> anyhow::Result<(bool, u32)> {
    let value = call_tile!(add_one, value);

    if value >= 10 {
        Ok((true, value))
    } else {
        Ok((false, value))
    }
}

#[sequence(kind = recur)]
fn fallible_add_sequence_until_goal(current: u64, goal: u64) -> anyhow::Result<(bool, u64)> {
    if current >= goal {
        Ok((true, current))
    } else {
        Ok((false, current + 1))
    }
}

#[sequence(kind = recur)]
fn fallible_sequence_step_that_fails(value: u32) -> anyhow::Result<(bool, u32)> {
    if value >= 2 {
        anyhow::bail!("sequence step failed");
    }

    Ok((false, call_tile!(add_one, value)))
}

#[test]
fn call_tile_invokes_tile_function() {
    assert_eq!(call_tile!(add_one, 41), 42);
}

#[test]
fn call_seq_invokes_sequence_function() {
    assert_eq!(call_seq!(add_sequence, 41), 42);
}

#[test]
fn external_macro_creates_typed_reference() {
    let external = external!("seed");
    let external: External<u64> = external;

    assert_eq!(external.name(), "seed");
    assert_eq!(external.into_ref().name(), "seed");
}

#[test]
fn auth_read_macro_invokes_typed_source() {
    let source = TestAuthSource;
    let value = auth_read!(&source, TestAuthRequest { fail: false }).expect("auth read");

    assert_eq!(value, 42);
}

#[test]
fn auth_read_macro_propagates_errors() {
    let source = TestAuthSource;
    let error =
        auth_read!(&source, TestAuthRequest { fail: true }).expect_err("auth read should fail");

    assert!(error.to_string().contains("auth read failed"));
}

#[test]
fn fallible_recursive_tile_macro_runs_until_done() {
    let value = call_recur_tile!(fallible_double_until_at_least_ten, 2).expect("recursive tile");

    assert_eq!(value, 16);
}

#[test]
fn fallible_recursive_tile_macro_accepts_context() {
    let value = call_recur_tile!(fallible_count_until_context, 0, 3).expect("recursive tile");

    assert_eq!(value, 3);
}

#[test]
fn fallible_recursive_tile_macro_propagates_errors() {
    let error =
        call_recur_tile!(fallible_step_that_fails, 0).expect_err("recursive tile should fail");

    assert!(error.to_string().contains("step failed"));
}

#[test]
fn fallible_recursive_sequence_macro_runs_until_done() {
    let value =
        call_recur_seq!(fallible_add_sequence_until_at_least_ten, 7).expect("recursive sequence");

    assert_eq!(value, 10);
}

#[test]
fn fallible_recursive_sequence_macro_accepts_context() {
    let value =
        call_recur_seq!(fallible_add_sequence_until_goal, 0, 3).expect("recursive sequence");

    assert_eq!(value, 3);
}

#[test]
fn fallible_recursive_sequence_macro_propagates_errors() {
    let error = call_recur_seq!(fallible_sequence_step_that_fails, 0)
        .expect_err("recursive sequence should fail");

    assert!(error.to_string().contains("sequence step failed"));
}

#[test]
fn tile_invocation_counter_counts_tiles_and_sequences_only_while_enabled() {
    assert_eq!(call_tile!(add_one, 1), 2);
    assert_eq!(super::stop_tile_invocation_counting(), None);

    super::start_tile_invocation_counting();
    assert_eq!(call_tile!(add_one, 1), 2);
    assert_eq!(call_seq!(add_sequence, 1), 2);
    assert_eq!(
        call_recur_tile!(fallible_double_until_at_least_ten, 2).expect("recursive tile"),
        16
    );
    assert_eq!(
        call_recur_seq!(fallible_add_sequence_until_goal, 0, 3).expect("recursive sequence"),
        3
    );

    assert_eq!(super::stop_tile_invocation_counting(), Some(11));
    assert_eq!(super::stop_tile_invocation_counting(), None);
}

#[test]
fn tile_invocation_counter_counts_fallible_failed_step() {
    super::start_tile_invocation_counting();

    let error =
        call_recur_tile!(fallible_step_that_fails, 0).expect_err("recursive tile should fail");

    assert!(error.to_string().contains("step failed"));
    assert_eq!(super::stop_tile_invocation_counting(), Some(3));
}

#[test]
fn tile_invocation_counter_counts_fallible_failed_sequence_step() {
    super::start_tile_invocation_counting();

    let error = call_recur_seq!(fallible_sequence_step_that_fails, 0)
        .expect_err("recursive sequence should fail");

    assert!(error.to_string().contains("sequence step failed"));
    assert_eq!(super::stop_tile_invocation_counting(), Some(5));
}
