//! Procedural attribute macros for the raster DSL.
//!
//! This crate intentionally stays tiny. Rust requires `#[proc_macro_attribute]`
//! definitions to live in a dedicated `proc-macro` crate, so the main DSL API
//! remains in `raster-inference::dsl` and re-exports these attributes.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn sequence(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    TokenStream::from(quote! {
        #item_fn
    })
}

#[proc_macro_attribute]
pub fn tile(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    TokenStream::from(quote! {
        #item_fn
    })
}
