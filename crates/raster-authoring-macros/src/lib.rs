use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn sequence(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

#[proc_macro_attribute]
pub fn tile(attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_fn = parse_macro_input!(item as ItemFn);

    let _recursive_tile = is_recursive_tile(attr);

    TokenStream::from(quote! {
        #item_fn
    })
}

fn is_recursive_tile(attr: TokenStream) -> bool {
    let attr = attr.to_string();

    attr.split(',')
        .map(|part| part.replace([' ', '"'], ""))
        .any(|part| part == "kind=recursive" || part == "kind=recur")
}
