//! mlir-tblgen execution wrapper.

use crate::Error;
use regex::Regex;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::LazyLock,
};

/// What a TableGen file contains, detected via text analysis.
#[derive(Debug, Clone, Default)]
pub struct TdFileContents {
    /// File contains a Dialect definition.
    pub has_dialect: bool,
    /// File contains Op definitions.
    pub has_ops: bool,
    /// File contains TypeDef definitions.
    pub has_types: bool,
    /// File contains AttrDef definitions.
    pub has_attrs: bool,
    /// File contains enum definitions.
    pub has_enums: bool,
    /// File uses FunctionOpInterface.
    pub has_function_interface: bool,
}

impl TdFileContents {
    /// Returns true if the file contains any definitions.
    pub fn has_any(&self) -> bool {
        self.has_dialect || self.has_ops || self.has_types || self.has_attrs || self.has_enums
    }
}

/// Tracks which TD file stems generated which content types.
///
/// This is used to generate correct include paths in the C++ registration file,
/// since different content types may come from different TD files with different stems.
#[derive(Debug, Clone, Default)]
pub struct GeneratedFiles {
    /// TD file stem that generated the dialect (e.g., "BrilOps" from BrilOps.td)
    pub dialect_stem: Option<String>,
    /// TD file stem that generated the ops
    pub ops_stem: Option<String>,
    /// TD file stem that generated the types
    pub types_stem: Option<String>,
    /// TD file stem that generated the attrs
    pub attrs_stem: Option<String>,
    /// TD file stem that generated the enums
    pub enums_stem: Option<String>,
    /// Whether FunctionOpInterface is used
    pub use_function_interface: bool,
}

// Static regexes for TD file content detection (compiled once)
static DIALECT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"def\s+\w+\s*:\s*Dialect\s*\{").unwrap());
static OP_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"def\s+\w+\s*:\s*\w*_?Op<").unwrap());
static TYPEDEF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"def\s+\w+\s*:\s*(\w*_?Type<|TypeDef<)").unwrap());
static ATTRDEF_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"def\s+\w+\s*:\s*(\w*_?Attr<|AttrDef<)").unwrap());
static ENUM_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(EnumAttr|IntEnumAttr|BitEnumAttr)").unwrap());

/// Detect what definitions a TableGen file contains.
///
/// This uses simple regex matching to detect:
/// - Dialect definitions: `def.*: Dialect`
/// - Op definitions: `def SomeName : SomeClass_Op<` or `def SomeName : Op<`
/// - Type definitions: `def SomeName : TypeDef<` or `def SomeName : SomeClass_Type<`
/// - Attr definitions: `def SomeName : AttrDef<` or `def SomeName : SomeClass_Attr<`
/// - Enum definitions: `EnumAttr` or `IntEnumAttr`
/// - FunctionOpInterface usage
///
/// Note: This distinguishes between `class` statements (base class definitions)
/// and `def` statements (actual definitions). Only `def` statements count as
/// defining ops/types/attrs.
pub fn detect_td_contents(path: &Path) -> Result<TdFileContents, Error> {
    let content = fs::read_to_string(path)?;

    Ok(TdFileContents {
        has_dialect: DIALECT_RE.is_match(&content),
        has_ops: OP_RE.is_match(&content),
        has_types: TYPEDEF_RE.is_match(&content),
        has_attrs: ATTRDEF_RE.is_match(&content),
        has_enums: ENUM_RE.is_match(&content),
        has_function_interface: content.contains("FunctionOpInterface"),
    })
}

/// Runner for mlir-tblgen commands.
pub struct TblgenRunner {
    /// Path to the mlir-tblgen binary
    tblgen_path: PathBuf,
    /// LLVM include directory
    llvm_include: PathBuf,
}

impl TblgenRunner {
    /// Create a new TblgenRunner from the LLVM prefix.
    pub fn new(llvm_prefix: &Path) -> Result<Self, Error> {
        let tblgen_path = llvm_prefix.join("bin").join("mlir-tblgen");

        if !tblgen_path.exists() {
            return Err(Error::TblgenNotFound(tblgen_path));
        }

        Ok(Self {
            tblgen_path,
            llvm_include: llvm_prefix.join("include"),
        })
    }

    /// Generate .inc files for a TD file based on its detected contents.
    ///
    /// Output file names are based on the TD file stem (e.g., `BrilOps.td` produces
    /// `BrilOpsDialect.h.inc`, `BrilOps.h.inc`, etc.), matching MLIR convention.
    pub fn generate_for_file(
        &self,
        td_file: &Path,
        include_dirs: &[PathBuf],
        output_dir: &Path,
        dialect_name: &str,
        contents: &TdFileContents,
    ) -> Result<(), Error> {
        // Use TD file stem for output naming (MLIR convention)
        let stem = td_file
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| {
                Error::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Invalid TD file path: {}", td_file.display()),
                ))
            })?;

        if contents.has_dialect {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Dialect.h.inc", stem)),
                "-gen-dialect-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Dialect.cpp.inc", stem)),
                "-gen-dialect-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_ops {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}.h.inc", stem)),
                "-gen-op-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}.cpp.inc", stem)),
                "-gen-op-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_types {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Types.h.inc", stem)),
                "-gen-typedef-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Types.cpp.inc", stem)),
                "-gen-typedef-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_attrs {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Attrs.h.inc", stem)),
                "-gen-attrdef-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Attrs.cpp.inc", stem)),
                "-gen-attrdef-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_enums {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Enums.h.inc", stem)),
                "-gen-enum-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Enums.cpp.inc", stem)),
                "-gen-enum-defs",
                Some(dialect_name),
            )?;
        }

        Ok(())
    }

    fn run_tblgen(
        &self,
        td_file: &Path,
        include_dirs: &[PathBuf],
        output: &Path,
        action: &str,
        dialect: Option<&str>,
    ) -> Result<(), Error> {
        let mut cmd = Command::new(&self.tblgen_path);
        cmd.arg(action).arg(td_file).arg("-o").arg(output);
        cmd.arg("-I").arg(&self.llvm_include);
        for include_dir in include_dirs {
            cmd.arg("-I").arg(include_dir);
        }
        if let Some(dialect_name) = dialect {
            cmd.arg(format!("--dialect={}", dialect_name));
        }

        let output_result = cmd.output()?;

        if !output_result.status.success() {
            let stderr = String::from_utf8_lossy(&output_result.stderr);
            return Err(Error::TblgenFailed(format!(
                "mlir-tblgen {} failed:\n{}",
                action, stderr
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::to_class_name;
    use std::io::Write;

    #[test]
    fn test_to_class_name() {
        assert_eq!(to_class_name("toy"), "Toy");
        assert_eq!(to_class_name("math_ext"), "MathExt");
        assert_eq!(to_class_name("my_custom_dialect"), "MyCustomDialect");
        assert_eq!(to_class_name(""), "");
    }

    #[test]
    fn test_detect_dialect() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_dialect.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def Bril_Dialect : Dialect {{
    let name = "bril";
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(contents.has_dialect);
        assert!(!contents.has_ops);
        assert!(!contents.has_types);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_ops() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_ops.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def Bril_AddOp : Bril_Op<"add", [Pure]> {{
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(!contents.has_dialect);
        assert!(
            contents.has_ops,
            "Should detect ops using custom _Op< base class"
        );
        assert!(!contents.has_types);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_types() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_types.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def Bril_PtrType : Bril_Type<"Ptr", "ptr"> {{
    let mnemonic = "ptr";
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(!contents.has_dialect);
        assert!(!contents.has_ops);
        assert!(
            contents.has_types,
            "Should detect types using custom _Type< base class"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_combined() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_combined.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def Bril_Dialect : Dialect {{
    let name = "bril";
}}

def Bril_PtrType : Bril_Type<"Ptr", "ptr"> {{
    let mnemonic = "ptr";
}}

def Bril_AddOp : Bril_Op<"add", [Pure]> {{
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(contents.has_dialect);
        assert!(contents.has_ops);
        assert!(contents.has_types);
        assert!(contents.has_any());

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_direct_op() {
        // Test detection of direct Op< usage (not via custom base class)
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_direct_op.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def MyOp : Op<MyDialect, "my_op"> {{
    let results = (outs I64);
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(contents.has_ops, "Should detect direct Op< usage");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_function_interface() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_func_interface.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
include "mlir/Interfaces/FunctionInterfaces.td"

def Bril_FuncOp : Bril_Op<"func", [
    FunctionOpInterface,
    IsolatedFromAbove
]> {{
    let arguments = (ins);
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(
            contents.has_function_interface,
            "Should detect FunctionOpInterface usage"
        );
        assert!(contents.has_ops, "Should also detect ops");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_detect_no_function_interface() {
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_no_func_interface.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def Bril_AddOp : Bril_Op<"add", [Pure]> {{
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}}
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(
            !contents.has_function_interface,
            "Should NOT detect FunctionOpInterface when not present"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_class_definitions_not_detected_as_ops() {
        // Test that base class definitions (using `class`) are NOT detected as ops.
        // Only actual op definitions (using `def`) should count.
        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_class_not_op.td");
        let mut file = std::fs::File::create(&path).unwrap();
        writeln!(
            file,
            r#"
def Bril_Dialect : Dialect {{
    let name = "bril";
}}

// This is a base class definition, NOT an op
class Bril_Op<string mnemonic, list<Trait> traits = []> :
    Op<Bril_Dialect, mnemonic, traits>;

// Similarly, base class for types should not count
class Bril_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Bril_Dialect, name, traits>;
"#
        )
        .unwrap();

        let contents = detect_td_contents(&path).unwrap();
        assert!(contents.has_dialect, "Should detect dialect");
        assert!(
            !contents.has_ops,
            "Should NOT detect ops from class definitions"
        );
        assert!(
            !contents.has_types,
            "Should NOT detect types from class definitions"
        );

        std::fs::remove_file(&path).ok();
    }
}
