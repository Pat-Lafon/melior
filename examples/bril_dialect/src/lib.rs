//! Bril dialect example demonstrating melior-build.
//!
//! This crate shows how to use melior-build with a complete dialect
//! based on the Bril intermediate language.
//!
//! Bril (Big Red Intermediate Language) is a simple IR for teaching compilers.
//! See: https://capra.cs.cornell.edu/bril/
//!
//! # File Structure
//!
//! This example demonstrates both single-file and multi-file TableGen approaches:
//!
//! ```text
//! src/dialect/
//! ├── BrilDialect.td           # Combined single-file (for melior::dialect! macro)
//! └── bril/
//!     ├── BrilDialect.td       # Split: dialect definition only
//!     ├── BrilTypes.td         # Split: type definitions
//!     └── BrilOps.td           # Split: operation definitions
//! ```
//!
//! # Why Two Approaches?
//!
//! - **`melior-build`** (in `build.rs`) uses the official LLVM `mlir-tblgen` tool,
//!   which fully supports TableGen's include mechanism. It can process split files
//!   and auto-detects what each file contains.
//!
//! - **`melior::dialect!`** macro uses a Rust-based TableGen parser that runs at
//!   compile time. This parser has limitations with include path resolution, so
//!   it works most reliably with a single combined TD file.
//!
//! Both tools serve different purposes:
//! - `melior-build` generates C++ code for dialect registration with MLIR
//! - `melior::dialect!` generates Rust wrapper types for operations

use melior::{Context, dialect::DialectRegistry, utility::register_all_dialects};

// Generate Rust operation wrappers from TableGen using the dialect! macro.
//
// We use the combined single-file BrilDialect.td here because the dialect! macro's
// TableGen parser works best with a single file. The split files in bril/ are used
// by melior-build (see build.rs) which uses the official mlir-tblgen tool.
melior::dialect! {
    name: "bril",
    td_file: "./examples/bril_dialect/src/dialect/BrilDialect.td",
}

// Include the generated registration code from melior-build.
// This provides: dialect_handle(), register(), load(), insert_into_registry()
include!(concat!(env!("OUT_DIR"), "/bril_register.rs"));

/// Create a context with the Bril dialect loaded.
pub fn create_context_with_bril() -> Context {
    let context = Context::new();

    // Register all built-in dialects
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();

    // Load the Bril dialect
    load(&context);

    context
}

#[cfg(test)]
mod tests {
    use super::*;
    use melior::ir::{
        attribute::{IntegerAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{IntegerType, FunctionType},
        Attribute, Block, BlockLike, Identifier, Location, Module, Region, RegionLike, Type, TypeLike,
    };

    // ==========================================================================
    // Registration Tests
    // ==========================================================================

    #[test]
    fn test_bril_all_operations_registered() {
        let context = create_context_with_bril();

        // All 27 bril operations
        let operations = [
            // Arithmetic (5)
            "bril.const",
            "bril.add",
            "bril.sub",
            "bril.mul",
            "bril.div",
            // Comparison (5)
            "bril.eq",
            "bril.lt",
            "bril.gt",
            "bril.le",
            "bril.ge",
            // Logical (3)
            "bril.not",
            "bril.and",
            "bril.or",
            // Utility (4)
            "bril.id",
            "bril.undef",
            "bril.nop",
            "bril.print",
            // Memory (5)
            "bril.alloc",
            "bril.free",
            "bril.store",
            "bril.load",
            "bril.ptr_add",
            // Control flow (5)
            "bril.call",
            "bril.jmp",
            "bril.br",
            "bril.func",
            "bril.ret",
        ];

        let registered_count = operations
            .iter()
            .filter(|op| context.is_registered_operation(op))
            .count();

        assert_eq!(
            registered_count,
            operations.len(),
            "All {} bril operations should be registered",
            operations.len()
        );
    }

    #[test]
    fn test_unregistered_operation_not_found() {
        let context = create_context_with_bril();

        // Verify that non-existent operations are NOT registered
        assert!(
            !context.is_registered_operation("bril.nonexistent"),
            "bril.nonexistent should NOT be registered"
        );
        assert!(
            !context.is_registered_operation("bril.fake_op"),
            "bril.fake_op should NOT be registered"
        );
        // Verify typos are caught
        assert!(
            !context.is_registered_operation("bril.addd"),
            "bril.addd (typo) should NOT be registered"
        );
    }

    // ==========================================================================
    // Operation Creation Tests
    // ==========================================================================

    #[test]
    fn test_create_const_operation() {
        let context = create_context_with_bril();
        let location = Location::unknown(&context);
        let i64_type = IntegerType::new(&context, 64).into();

        let const_op = OperationBuilder::new("bril.const", location)
            .add_attributes(&[(
                Identifier::new(&context, "value"),
                IntegerAttribute::new(i64_type, 42).into(),
            )])
            .add_results(&[i64_type])
            .build()
            .unwrap();

        assert!(const_op.verify());
        assert_eq!(const_op.name().as_string_ref().as_str().unwrap(), "bril.const");
        assert_eq!(const_op.result_count(), 1);
    }

    #[test]
    fn test_create_add_operation() {
        let context = create_context_with_bril();
        let location = Location::unknown(&context);
        let i64_type = IntegerType::new(&context, 64).into();

        let block = Block::new(&[(i64_type, location), (i64_type, location)]);
        let lhs = block.argument(0).unwrap().into();
        let rhs = block.argument(1).unwrap().into();

        let add_op = OperationBuilder::new("bril.add", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[i64_type])
            .build()
            .unwrap();

        assert!(add_op.verify());
        assert_eq!(add_op.operand_count(), 2);
        assert_eq!(add_op.result_count(), 1);
    }

    #[test]
    fn test_create_nop_operation() {
        let context = create_context_with_bril();
        let location = Location::unknown(&context);

        let nop_op = OperationBuilder::new("bril.nop", location)
            .build()
            .unwrap();

        assert!(nop_op.verify());
        assert_eq!(nop_op.operand_count(), 0);
        assert_eq!(nop_op.result_count(), 0);
    }

    // ==========================================================================
    // Custom Type Tests (!bril.ptr<T>)
    // ==========================================================================

    #[test]
    fn test_parse_ptr_type() {
        let context = create_context_with_bril();

        let ptr_type = Type::parse(&context, "!bril.ptr<i64>");
        assert!(ptr_type.is_some());

        let ptr_type = ptr_type.unwrap();
        assert!(!ptr_type.is_integer());
        assert!(!ptr_type.is_index());
    }

    // ==========================================================================
    // dialect! Macro Output Tests
    // ==========================================================================

    #[test]
    fn test_dialect_macro_module_exists() {
        use crate::bril;

        // Macro generates: Bril_AddOp -> AddOperation, Bril_ConstantOp -> ConstantOperation
        assert_eq!(bril::AddOperation::name(), "bril.add");
        assert_eq!(bril::SubOperation::name(), "bril.sub");
        assert_eq!(bril::MulOperation::name(), "bril.mul");
        assert_eq!(bril::ConstantOperation::name(), "bril.const");
        assert_eq!(bril::NopOperation::name(), "bril.nop");
    }

    #[test]
    fn test_dialect_macro_operation_builders() {
        use crate::bril;

        let context = create_context_with_bril();
        let location = Location::unknown(&context);
        let i64_type: Type = IntegerType::new(&context, 64).into();

        let block = Block::new(&[(i64_type, location), (i64_type, location)]);
        let lhs = block.argument(0).unwrap().into();
        let rhs = block.argument(1).unwrap().into();

        let add_op = bril::AddOperation::builder(&context, location)
            .lhs(lhs)
            .rhs(rhs)
            .build();

        assert!(add_op.as_operation().verify());
    }

    #[test]
    fn test_dialect_macro_tryfrom() {
        use crate::bril;
        use std::convert::TryFrom;

        let context = create_context_with_bril();
        let location = Location::unknown(&context);
        let i64_type: Type = IntegerType::new(&context, 64).into();

        let block = Block::new(&[(i64_type, location), (i64_type, location)]);
        let lhs = block.argument(0).unwrap().into();
        let rhs = block.argument(1).unwrap().into();

        let raw_op = OperationBuilder::new("bril.add", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[i64_type])
            .build()
            .unwrap();

        let typed_op = bril::AddOperation::try_from(raw_op);
        assert!(typed_op.is_ok());
    }

    // ==========================================================================
    // End-to-End MLIR Module Tests
    // ==========================================================================

    #[test]
    fn test_parse_bril_with_ptr_type() {
        let context = create_context_with_bril();

        let mlir_source = r#"
            module {
                func.func @alloc_test(%size: i64) -> !bril.ptr<i64> {
                    %ptr = "bril.alloc"(%size) : (i64) -> !bril.ptr<i64>
                    return %ptr : !bril.ptr<i64>
                }
            }
        "#;

        let module = Module::parse(&context, mlir_source).unwrap();
        assert!(module.as_operation().verify());
    }

    #[test]
    fn test_build_module_programmatically() {
        let context = create_context_with_bril();
        let location = Location::unknown(&context);
        let i64_type: Type = IntegerType::new(&context, 64).into();

        let body_block = Block::new(&[(i64_type, location), (i64_type, location)]);
        let arg0 = body_block.argument(0).unwrap().into();
        let arg1 = body_block.argument(1).unwrap().into();

        let add_op = OperationBuilder::new("bril.add", location)
            .add_operands(&[arg0, arg1])
            .add_results(&[i64_type])
            .build()
            .unwrap();

        let add_result = add_op.result(0).unwrap().into();
        body_block.append_operation(add_op);

        let ret_op = OperationBuilder::new("func.return", location)
            .add_operands(&[add_result])
            .build()
            .unwrap();
        body_block.append_operation(ret_op);

        let func_region = Region::new();
        func_region.append_block(body_block);

        let func_type = FunctionType::new(&context, &[i64_type, i64_type], &[i64_type]);
        let func_op = OperationBuilder::new("func.func", location)
            .add_attributes(&[
                (
                    Identifier::new(&context, "sym_name"),
                    Attribute::parse(&context, "\"add_two\"").unwrap(),
                ),
                (
                    Identifier::new(&context, "function_type"),
                    TypeAttribute::new(func_type.into()).into(),
                ),
            ])
            .add_regions([func_region])
            .build()
            .unwrap();

        let module_block = Block::new(&[]);
        module_block.append_operation(func_op);

        let module_region = Region::new();
        module_region.append_block(module_block);

        let module_op = OperationBuilder::new("builtin.module", location)
            .add_regions([module_region])
            .build()
            .unwrap();

        let module = Module::from_operation(module_op).unwrap();
        assert!(module.as_operation().verify());

        let module_str = module.as_operation().to_string();
        assert!(module_str.contains("bril.add"));
        assert!(module_str.contains("func.func"));
    }

    #[test]
    fn test_module_with_multiple_bril_ops() {
        let context = create_context_with_bril();

        let mlir_source = r#"
            module {
                func.func @complex(%a: i64, %b: i64) -> i1 {
                    %sum = "bril.add"(%a, %b) : (i64, i64) -> i64
                    %diff = "bril.sub"(%a, %b) : (i64, i64) -> i64
                    %prod = "bril.mul"(%sum, %diff) : (i64, i64) -> i64
                    %cmp = "bril.lt"(%prod, %a) : (i64, i64) -> i1
                    return %cmp : i1
                }
            }
        "#;

        let module = Module::parse(&context, mlir_source).unwrap();
        assert!(module.as_operation().verify());
    }
}
