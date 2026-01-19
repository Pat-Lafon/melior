//! mlir-tblgen execution wrapper.

use crate::{Error, to_class_name};
use regex::Regex;
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// Options controlling what code to generate.
#[derive(Debug, Clone, Default)]
pub struct GenerationOptions {
    /// Generate custom type definitions.
    pub generate_types: bool,
    /// Generate custom attribute definitions.
    pub generate_attributes: bool,
    /// Generate enum definitions.
    pub generate_enums: bool,
    /// Include FunctionInterfaces header (for FunctionOpInterface support).
    pub use_function_interface: bool,
}

impl GenerationOptions {
    /// Create GenerationOptions from auto-detected file contents.
    pub fn from_detected(contents: &TdFileContents) -> Self {
        Self {
            generate_types: contents.has_types,
            generate_attributes: contents.has_attrs,
            generate_enums: contents.has_enums,
            use_function_interface: contents.has_function_interface,
        }
    }
}

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

/// Detect what definitions a TableGen file contains.
///
/// This uses simple regex matching to detect:
/// - Dialect definitions: `def.*: Dialect`
/// - Op definitions: `: SomeClass_Op<` or `: Op<`
/// - Type definitions: `TypeDef<` or `: SomeClass_Type<`
/// - Attr definitions: `AttrDef<`
/// - Enum definitions: `EnumAttr` or `IntEnumAttr`
/// - FunctionOpInterface usage
pub fn detect_td_contents(path: &Path) -> Result<TdFileContents, Error> {
    let content = fs::read_to_string(path)?;

    // Patterns to detect different definition types
    // These are intentionally broad to catch most cases
    //
    // Note: TableGen dialects often define base classes like `class Foo_Op<...> :
    // Op<...>` and then use them like `def Foo_AddOp : Foo_Op<"add">`. We need
    // to detect both:
    // 1. Direct usage: `: Op<`
    // 2. Custom base class usage: `: SomePrefix_Op<` (ops typically end with _Op)
    let dialect_re = Regex::new(r"def\s+\w+\s*:\s*Dialect\s*\{").unwrap();
    let op_re = Regex::new(r":\s*\w*_?Op<").unwrap();
    let typedef_re = Regex::new(r":\s*(\w*_?Type<|TypeDef<)").unwrap();
    let attrdef_re = Regex::new(r":\s*(\w*_?Attr<|AttrDef<)").unwrap();
    let enum_re = Regex::new(r"(EnumAttr|IntEnumAttr|BitEnumAttr)").unwrap();

    Ok(TdFileContents {
        has_dialect: dialect_re.is_match(&content),
        has_ops: op_re.is_match(&content),
        has_types: typedef_re.is_match(&content),
        has_attrs: attrdef_re.is_match(&content),
        has_enums: enum_re.is_match(&content),
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
    pub fn generate_for_file(
        &self,
        td_file: &Path,
        include_dirs: &[PathBuf],
        output_dir: &Path,
        dialect_name: &str,
        contents: &TdFileContents,
    ) -> Result<(), Error> {
        let class_name = to_class_name(dialect_name);

        if contents.has_dialect {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Dialect.h.inc", class_name)),
                "-gen-dialect-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Dialect.cpp.inc", class_name)),
                "-gen-dialect-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_ops {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Ops.h.inc", class_name)),
                "-gen-op-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Ops.cpp.inc", class_name)),
                "-gen-op-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_types {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Types.h.inc", class_name)),
                "-gen-typedef-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Types.cpp.inc", class_name)),
                "-gen-typedef-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_attrs {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Attrs.h.inc", class_name)),
                "-gen-attrdef-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Attrs.cpp.inc", class_name)),
                "-gen-attrdef-defs",
                Some(dialect_name),
            )?;
        }

        if contents.has_enums {
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Enums.h.inc", class_name)),
                "-gen-enum-decls",
                Some(dialect_name),
            )?;
            self.run_tblgen(
                td_file,
                include_dirs,
                &output_dir.join(format!("{}Enums.cpp.inc", class_name)),
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
}
