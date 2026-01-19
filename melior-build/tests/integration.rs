//! Integration tests for melior-build.

#[test]
fn test_dialect_builder_construction() {
    use melior_build::DialectBuilder;

    let _builder = DialectBuilder::new("test_dialect")
        .td_file("path/to/dialect.td")
        .include_dir("/usr/include")
        .cpp_namespace("mlir::test");
}

#[test]
fn test_dialect_builder_multiple_files() {
    use melior_build::DialectBuilder;

    let _builder = DialectBuilder::new("multi")
        .td_files(&["file1.td", "file2.td", "file3.td"])
        .include_dirs(&["/include1", "/include2"]);
}

#[test]
fn test_dialect_builder_cpp_files() {
    use melior_build::DialectBuilder;

    let _builder = DialectBuilder::new("my_dialect")
        .td_file("dialect.td")
        .cpp_file("src/Verifiers.cpp")
        .cpp_files(&["src/Canonicalize.cpp", "src/Builders.cpp"]);
}

#[test]
fn test_cpp_generation() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_dialect_capi.cpp");

    let options = melior_build::tblgen::GenerationOptions::default();
    melior_build::cpp_gen::generate_cpp_registration(
        "operand_test",
        "mlir::operand_test",
        &options,
        &output_path,
    )
    .unwrap();

    let content = std::fs::read_to_string(&output_path).unwrap();

    assert!(content.contains("OperandTestDialect.h.inc"));
    assert!(content.contains("OperandTestDialect.cpp.inc"));
    assert!(content.contains("OperandTestOps.h.inc"));
    assert!(content.contains("OperandTestOps.cpp.inc"));
    assert!(content.contains("mlir::operand_test::OperandTestDialect"));
    assert!(content.contains("MLIR_DEFINE_CAPI_DIALECT_REGISTRATION"));

    std::fs::remove_file(&output_path).ok();
}

#[test]
fn test_rust_ffi_generation() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("test_register.rs");

    melior_build::rust_gen::generate_rust_ffi("operand_test", &output_path).unwrap();

    let content = std::fs::read_to_string(&output_path).unwrap();

    assert!(content.contains("mlirGetDialectHandle__operand_test__"));
    assert!(content.contains("pub fn dialect_handle()"));
    assert!(content.contains("pub fn register("));
    assert!(content.contains("pub fn load("));
    assert!(content.contains("pub fn insert_into_registry("));
    assert!(content.contains("::melior::dialect::DialectHandle"));

    std::fs::remove_file(&output_path).ok();
}

#[test]
fn test_rust_ffi_syntax() {
    let temp_dir = std::env::temp_dir();
    let output_path = temp_dir.join("syntax_test.rs");

    melior_build::rust_gen::generate_rust_ffi("my_dialect", &output_path).unwrap();

    let content = std::fs::read_to_string(&output_path).unwrap();

    assert!(content.contains("mod my_dialect_registration"));
    assert!(content.contains("pub use my_dialect_registration::"));

    let open_braces = content.matches('{').count();
    let close_braces = content.matches('}').count();
    assert_eq!(open_braces, close_braces, "Braces should be balanced");

    std::fs::remove_file(&output_path).ok();
}
