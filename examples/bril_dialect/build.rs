//! Build script for the Bril dialect example.
//!
//! Demonstrates melior-build's multi-file support with auto-detection:
//! - `BrilDialect.td` - Dialect definition
//! - `BrilTypes.td` - Custom type definitions (PtrType)
//! - `BrilOps.td` - Operation definitions (including FuncOp with FunctionOpInterface)

use melior_build::DialectBuilder;

fn main() {
    DialectBuilder::new("bril")
        .td_file("src/dialect/bril/BrilDialect.td")
        .td_file("src/dialect/bril/BrilTypes.td")
        .td_file("src/dialect/bril/BrilOps.td")
        .include_dir("src/dialect")
        .cpp_namespace("mlir::bril")
        .build()
        .expect("Failed to build bril dialect");
}
