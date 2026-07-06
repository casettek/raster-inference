//! Gemma committed-external schemas and encoder (WS2).
//!
//! Layout constraint (WS1 catalog C33): tiles live in their own file;
//! this root only declares modules.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

#[cfg(feature = "encode")]
pub mod encode;
pub mod smoke;
pub mod types;
