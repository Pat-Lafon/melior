//! Build script for the custom dialect example.
//!
//! This demonstrates how to use melior-build to compile a custom MLIR dialect
//! from TableGen definitions.

use melior_build::DialectBuilder;

fn main() {
    // Build the math_ext dialect from TableGen
    DialectBuilder::new("math_ext")
        .td_file("src/dialect/MathDialect.td")
        .cpp_namespace("mlir::math_ext")
        .build()
        .expect("Failed to build math_ext dialect");

    println!("cargo:rerun-if-changed=src/dialect/MathDialect.td");
}
