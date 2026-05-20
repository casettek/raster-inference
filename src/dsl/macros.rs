#[macro_export]
macro_rules! call_tile {
    ($tile:ident $(,)?) => {
        {
            $crate::dsl::record_tile_invocation();
            $tile()
        }
    };
    ($tile:ident, $($args:expr),+ $(,)?) => {
        {
            $crate::dsl::record_tile_invocation();
            $tile($($args),+)
        }
    };
}

#[macro_export]
macro_rules! call_seq {
    ($sequence:ident $(,)?) => {
        {
            $crate::dsl::record_tile_invocation();
            $sequence()
        }
    };
    ($sequence:ident, $($args:expr),+ $(,)?) => {
        {
            $crate::dsl::record_tile_invocation();
            $sequence($($args),+)
        }
    };
}

#[macro_export]
macro_rules! call_recur_tile {
    ($tile:ident, ($state_a:expr, $state_b:expr) $(,)?) => {
        $crate::__dsl_run_recur_tile_result_pair!($tile, $state_a, $state_b)
    };
    ($tile:ident, ($state_a:expr, $state_b:expr), $($context:expr),+ $(,)?) => {
        $crate::__dsl_run_recur_tile_result_pair!($tile, $state_a, $state_b, $($context),+)
    };
    ($tile:ident, $state:expr $(,)?) => {
        $crate::__dsl_run_recur_tile_result!($tile, $state)
    };
    ($tile:ident, $state:expr, $($context:expr),+ $(,)?) => {
        $crate::__dsl_run_recur_tile_result!($tile, $state, $($context),+)
    };
}

#[macro_export]
macro_rules! call_recur_seq {
    ($sequence:ident, ($state_a:expr, $state_b:expr) $(,)?) => {
        $crate::__dsl_run_recur_sequence_result_pair!($sequence, $state_a, $state_b)
    };
    ($sequence:ident, ($state_a:expr, $state_b:expr), $($context:expr),+ $(,)?) => {
        $crate::__dsl_run_recur_sequence_result_pair!($sequence, $state_a, $state_b, $($context),+)
    };
    ($sequence:ident, $state:expr $(,)?) => {
        $crate::__dsl_run_recur_sequence_result!($sequence, $state)
    };
    ($sequence:ident, $state:expr, $($context:expr),+ $(,)?) => {
        $crate::__dsl_run_recur_sequence_result!($sequence, $state, $($context),+)
    };
}

#[macro_export]
macro_rules! external {
    ($name:literal) => {
        $crate::dsl::external($name)
    };
    ($name:expr) => {
        $crate::dsl::external($name)
    };
}

#[macro_export]
macro_rules! auth_read {
    ($source:expr, $request:expr $(,)?) => {
        $crate::shared::artifacts::artifact_io::ArtifactIo::auth_read($source, $request)
    };
}

#[macro_export]
macro_rules! __dsl_run_recur_tile {
    ($tile:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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
macro_rules! __dsl_run_recur_tile_result {
    ($tile:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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
macro_rules! __dsl_run_recur_tile_result_pair {
    ($tile:ident, $state_a:expr, $state_b:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        loop {
            $crate::dsl::record_tile_invocation();
            let (done, next_state_a, next_state_b) = match $tile(state_a, state_b) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok((next_state_a, next_state_b));
            }
            state_a = next_state_a;
            state_b = next_state_b;
        }
    }};
    ($tile:ident, $state_a:expr, $state_b:expr, $($context:expr),+ $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        loop {
            $crate::dsl::record_tile_invocation();
            let (done, next_state_a, next_state_b) = match $tile(state_a, state_b, $($context),+) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok((next_state_a, next_state_b));
            }
            state_a = next_state_a;
            state_b = next_state_b;
        }
    }};
}

#[macro_export]
macro_rules! __dsl_run_recur_sequence {
    ($sequence:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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
            $crate::dsl::record_tile_invocation();
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

#[macro_export]
macro_rules! __dsl_run_recur_sequence_result {
    ($sequence:ident, $state:expr $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::dsl::record_tile_invocation();
            let (done, next_state) = match $sequence(state) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok(next_state);
            }
            state = next_state;
        }
    }};
    ($sequence:ident, $state:expr, $($context:expr),+ $(,)?) => {{
        let mut state = $state;
        loop {
            $crate::dsl::record_tile_invocation();
            let (done, next_state) = match $sequence(state, $($context),+) {
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
macro_rules! __dsl_run_recur_sequence_result_pair {
    ($sequence:ident, $state_a:expr, $state_b:expr $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        loop {
            $crate::dsl::record_tile_invocation();
            let (done, next_state_a, next_state_b) = match $sequence(state_a, state_b) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok((next_state_a, next_state_b));
            }
            state_a = next_state_a;
            state_b = next_state_b;
        }
    }};
    ($sequence:ident, $state_a:expr, $state_b:expr, $($context:expr),+ $(,)?) => {{
        let mut state_a = $state_a;
        let mut state_b = $state_b;
        loop {
            $crate::dsl::record_tile_invocation();
            let (done, next_state_a, next_state_b) = match $sequence(state_a, state_b, $($context),+) {
                Ok(next) => next,
                Err(error) => break Err(error),
            };
            if done {
                break Ok((next_state_a, next_state_b));
            }
            state_a = next_state_a;
            state_b = next_state_b;
        }
    }};
}
