use std::{cell::Cell, marker::PhantomData};

pub use raster_authoring_macros::{sequence, tile};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExternalRef {
    name: String,
}

impl ExternalRef {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct External<T> {
    reference: ExternalRef,
    marker: PhantomData<fn() -> T>,
}

impl<T> External<T> {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            reference: ExternalRef::new(name),
            marker: PhantomData,
        }
    }

    pub fn name(&self) -> &str {
        self.reference.name()
    }

    pub fn into_ref(self) -> ExternalRef {
        self.reference
    }
}

pub fn external<T>(name: impl Into<String>) -> External<T> {
    External::new(name)
}

pub trait AuthRead<Request> {
    type Output;

    fn auth_read(&self, request: Request) -> anyhow::Result<Self::Output>;
}

pub fn auth_read<Source, Request>(
    source: &Source,
    request: Request,
) -> anyhow::Result<<Source as AuthRead<Request>>::Output>
where
    Source: AuthRead<Request> + ?Sized,
{
    source.auth_read(request)
}

thread_local! {
    static TILE_INVOCATION_COUNT: Cell<Option<u64>> = const { Cell::new(None) };
}

pub fn start_tile_invocation_counting() {
    TILE_INVOCATION_COUNT.with(|count| count.set(Some(0)));
}

pub fn stop_tile_invocation_counting() -> Option<u64> {
    TILE_INVOCATION_COUNT.with(|count| {
        let total = count.get();
        count.set(None);
        total
    })
}

#[inline(always)]
pub fn record_tile_invocation() {
    TILE_INVOCATION_COUNT.with(|count| {
        if let Some(total) = count.get() {
            count.set(Some(total.saturating_add(1)));
        }
    });
}

pub mod prelude {
    pub use crate::{
        auth_read, call_recur_seq, call_recur_tile, call_recur_tile_result, call_seq, call_tile,
        external, raster_authoring::sequence, raster_authoring::tile, raster_authoring::AuthRead,
        raster_authoring::External, raster_authoring::ExternalRef,
    };
}

#[macro_export]
macro_rules! call_tile {
    ($tile:ident $(,)?) => {
        {
            $crate::raster_authoring::record_tile_invocation();
            $tile()
        }
    };
    ($tile:ident, $($args:expr),+ $(,)?) => {
        {
            $crate::raster_authoring::record_tile_invocation();
            $tile($($args),+)
        }
    };
}

#[macro_export]
macro_rules! call_seq {
    ($sequence:ident $(,)?) => {
        {
            $crate::raster_authoring::record_tile_invocation();
            $sequence()
        }
    };
    ($sequence:ident, $($args:expr),+ $(,)?) => {
        {
            $crate::raster_authoring::record_tile_invocation();
            $sequence($($args),+)
        }
    };
}

#[macro_export]
macro_rules! call_recur_tile {
    ($tile:ident $(,)?) => {
        $crate::__raster_authoring_run_recur_tile!($tile)
    };
    ($tile:ident, $($args:expr),+ $(,)?) => {
        $crate::__raster_authoring_run_recur_tile!($tile, $($args),+)
    };
}

#[macro_export]
macro_rules! call_recur_tile_result {
    ($tile:ident, $state:expr $(,)?) => {
        $crate::__raster_authoring_run_recur_tile_result!($tile, $state)
    };
    ($tile:ident, $state:expr, $($context:expr),+ $(,)?) => {
        $crate::__raster_authoring_run_recur_tile_result!($tile, $state, $($context),+)
    };
}

#[macro_export]
macro_rules! call_recur_seq {
    ($sequence:ident $(,)?) => {
        $crate::__raster_authoring_run_recur_sequence!($sequence)
    };
    ($sequence:ident, $($args:expr),+ $(,)?) => {
        $crate::__raster_authoring_run_recur_sequence!($sequence, $($args),+)
    };
}

#[macro_export]
macro_rules! external {
    ($name:literal) => {
        $crate::raster_authoring::external($name)
    };
    ($name:expr) => {
        $crate::raster_authoring::external($name)
    };
}

#[macro_export]
macro_rules! auth_read {
    ($source:expr, $request:expr $(,)?) => {
        $crate::raster_authoring::auth_read($source, $request)
    };
}

#[macro_export]
macro_rules! __raster_authoring_run_recur_tile {
    ($tile:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state) = $tile(state);
            if done {
                break next_state;
            }
            state = next_state;
        }
    }};
    ($tile:ident, $state_a:expr, $state_b:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state_a, next_state_b) = $tile(state_a, state_b);
            if done {
                break (next_state_a, next_state_b);
            }
            state_a = next_state_a;
            state_b = next_state_b;
        }
    }};
    ($tile:ident, $state_a:expr, $state_b:expr, $state_c:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        let mut state_c = $state_c;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state_a, next_state_b, next_state_c) = $tile(state_a, state_b, state_c);
            if done {
                break (next_state_a, next_state_b, next_state_c);
            }
            state_a = next_state_a;
            state_b = next_state_b;
            state_c = next_state_c;
        }
    }};
    ($tile:ident, $state_a:expr, $state_b:expr, $state_c:expr, $state_d:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        let mut state_c = $state_c;
        let mut state_d = $state_d;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state_a, next_state_b, next_state_c, next_state_d) =
                $tile(state_a, state_b, state_c, state_d);
            if done {
                break (next_state_a, next_state_b, next_state_c, next_state_d);
            }
            state_a = next_state_a;
            state_b = next_state_b;
            state_c = next_state_c;
            state_d = next_state_d;
        }
    }};
}

#[macro_export]
macro_rules! __raster_authoring_run_recur_tile_result {
    ($tile:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state) = match $tile(state) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok(next_state);
            }
            state = next_state;
        }
    }};
    ($tile:ident, $state:expr, $($context:expr),+ $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state) = match $tile(state, $($context),+) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok(next_state);
            }
            state = next_state;
        }
    }};
}

#[macro_export]
macro_rules! __raster_authoring_run_recur_sequence {
    ($sequence:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state) = $sequence(state);
            if done {
                break next_state;
            }
            state = next_state;
        }
    }};
    ($sequence:ident, $state_a:expr, $state_b:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state_a, next_state_b) = $sequence(state_a, state_b);
            if done {
                break (next_state_a, next_state_b);
            }
            state_a = next_state_a;
            state_b = next_state_b;
        }
    }};
    ($sequence:ident, $state_a:expr, $state_b:expr, $state_c:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        let mut state_c = $state_c;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state_a, next_state_b, next_state_c) =
                $sequence(state_a, state_b, state_c);
            if done {
                break (next_state_a, next_state_b, next_state_c);
            }
            state_a = next_state_a;
            state_b = next_state_b;
            state_c = next_state_c;
        }
    }};
    ($sequence:ident, $state_a:expr, $state_b:expr, $state_c:expr, $state_d:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        let mut state_c = $state_c;
        let mut state_d = $state_d;
        loop {
            $crate::raster_authoring::record_tile_invocation();
            let (done, next_state_a, next_state_b, next_state_c, next_state_d) =
                $sequence(state_a, state_b, state_c, state_d);
            if done {
                break (next_state_a, next_state_b, next_state_c, next_state_d);
            }
            state_a = next_state_a;
            state_b = next_state_b;
            state_c = next_state_c;
            state_d = next_state_d;
        }
    }};
}

#[cfg(test)]
mod tests {
    use crate::raster_authoring::prelude::*;

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
    fn count_to(current: u64, goal: u64) -> (bool, u64, u64) {
        if current >= goal {
            (true, current, goal)
        } else {
            (false, current + 1, goal)
        }
    }

    #[tile(kind = recur)]
    fn double_until_at_least_ten(value: u32) -> (bool, u32) {
        if value >= 10 {
            (true, value)
        } else {
            (false, value * 2)
        }
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
    fn add_sequence_until_at_least_ten(value: u32) -> (bool, u32) {
        let value = call_tile!(add_one, value);

        if value >= 10 {
            (true, value)
        } else {
            (false, value)
        }
    }

    #[sequence(kind = recur)]
    fn add_sequence_until_goal(current: u64, goal: u64) -> (bool, u64, u64) {
        if current >= goal {
            (true, current, goal)
        } else {
            (false, current + 1, goal)
        }
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
    fn recursive_tile_macro_runs_until_done_for_single_state() {
        assert_eq!(call_recur_tile!(double_until_at_least_ten, 2), 16);
    }

    #[test]
    fn recursive_tile_macro_runs_until_done_for_tuple_state() {
        assert_eq!(call_recur_tile!(count_to, 0, 3), (3, 3));
    }

    #[test]
    fn recursive_sequence_macro_runs_until_done_for_single_state() {
        assert_eq!(call_recur_seq!(add_sequence_until_at_least_ten, 7), 10);
    }

    #[test]
    fn recursive_sequence_macro_runs_until_done_for_tuple_state() {
        assert_eq!(call_recur_seq!(add_sequence_until_goal, 0, 3), (3, 3));
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
        let value =
            call_recur_tile_result!(fallible_double_until_at_least_ten, 2).expect("recursive tile");

        assert_eq!(value, 16);
    }

    #[test]
    fn fallible_recursive_tile_macro_accepts_context() {
        let value =
            call_recur_tile_result!(fallible_count_until_context, 0, 3).expect("recursive tile");

        assert_eq!(value, 3);
    }

    #[test]
    fn fallible_recursive_tile_macro_propagates_errors() {
        let error = call_recur_tile_result!(fallible_step_that_fails, 0)
            .expect_err("recursive tile should fail");

        assert!(error.to_string().contains("step failed"));
    }

    #[test]
    fn tile_invocation_counter_counts_tiles_and_sequences_only_while_enabled() {
        assert_eq!(call_tile!(add_one, 1), 2);
        assert_eq!(super::stop_tile_invocation_counting(), None);

        super::start_tile_invocation_counting();
        assert_eq!(call_tile!(add_one, 1), 2);
        assert_eq!(call_seq!(add_sequence, 1), 2);
        assert_eq!(call_recur_tile!(double_until_at_least_ten, 2), 16);
        assert_eq!(call_recur_seq!(add_sequence_until_goal, 0, 3), (3, 3));

        assert_eq!(super::stop_tile_invocation_counting(), Some(11));
        assert_eq!(super::stop_tile_invocation_counting(), None);
    }

    #[test]
    fn tile_invocation_counter_counts_fallible_failed_step() {
        super::start_tile_invocation_counting();

        let error = call_recur_tile_result!(fallible_step_that_fails, 0)
            .expect_err("recursive tile should fail");

        assert!(error.to_string().contains("step failed"));
        assert_eq!(super::stop_tile_invocation_counting(), Some(3));
    }
}
