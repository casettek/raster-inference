use std::marker::PhantomData;

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

pub mod prelude {
    pub use crate::{
        call_recur_seq, call_recur_tile, call_seq, call_tile, external, raster_authoring::sequence,
        raster_authoring::tile, raster_authoring::External, raster_authoring::ExternalRef,
    };
}

#[macro_export]
macro_rules! call_tile {
    ($tile:ident $(,)?) => {
        $tile()
    };
    ($tile:ident, $($args:expr),+ $(,)?) => {
        $tile($($args),+)
    };
}

#[macro_export]
macro_rules! call_seq {
    ($sequence:ident $(,)?) => {
        $sequence()
    };
    ($sequence:ident, $($args:expr),+ $(,)?) => {
        $sequence($($args),+)
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
macro_rules! __raster_authoring_run_recur_tile {
    ($tile:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
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
macro_rules! __raster_authoring_run_recur_sequence {
    ($sequence:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
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
}
