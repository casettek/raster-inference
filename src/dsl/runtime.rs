use std::{cell::Cell, marker::PhantomData};

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
pub fn record_tile_invocation(invocation_kind: &str, name: &str) {
    TILE_INVOCATION_COUNT.with(|count| {
        if let Some(total) = count.get() {
            let next_total = total.saturating_add(1);
            count.set(Some(next_total));
            crate::trace::tile_invoked(invocation_kind, name, next_total);
        }
    });
}
