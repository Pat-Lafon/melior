//! Custom dialect example demonstrating melior-build.
//!
//! This crate shows how to define a custom MLIR dialect using TableGen
//! and register it with melior using melior-build.

use melior::{Context, dialect::DialectRegistry, utility::register_all_dialects};

// Generate Rust operation wrappers from TableGen using the dialect! macro
// Note: paths are relative to the workspace root
melior::dialect! {
    name: "math_ext",
    files: ["examples/custom_dialect/src/dialect/MathDialect.td"],
}

// Include the generated registration code from melior-build.
// This provides: dialect_handle(), register(), load(), insert_into_registry()
include!(concat!(env!("OUT_DIR"), "/math_ext_register.rs"));

/// Create a context with the math_ext dialect loaded.
pub fn create_context_with_math_ext() -> Context {
    let context = Context::new();

    // Register all built-in dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Load our custom dialect (function from included file)
    // Using load() instead of register() ensures the dialect is fully initialized
    load(&context);

    context
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dialect_registration() {
        let context = create_context_with_math_ext();

        // Verify the dialect is registered
        assert!(
            context.is_registered_operation("math_ext.add"),
            "math_ext.add should be registered"
        );
        assert!(
            context.is_registered_operation("math_ext.mul"),
            "math_ext.mul should be registered"
        );
        assert!(
            context.is_registered_operation("math_ext.const"),
            "math_ext.const should be registered"
        );
    }

    #[test]
    fn test_dialect_load() {
        let context = create_context_with_math_ext();

        // Load the dialect and verify it's valid
        let dialect = context.get_or_load_dialect("math_ext");
        assert!(
            !dialect
                .namespace()
                .expect("namespace should be valid UTF-8")
                .is_empty(),
            "Dialect namespace should not be empty"
        );
    }

    #[test]
    fn test_create_module() {
        use melior::ir::{Location, Module};

        let context = create_context_with_math_ext();
        let location = Location::unknown(&context);
        let _module = Module::new(location);

        // The math_ext module from the dialect! macro should have operation builders
        // This test verifies the generated code compiles
    }
}
