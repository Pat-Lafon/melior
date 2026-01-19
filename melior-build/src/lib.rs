//! Build-time helpers for registering custom MLIR dialects with melior.
//!
//! This crate provides utilities for compiling TableGen dialect definitions
//! into C++ code that can be linked with your Rust application, enabling
//! proper dialect registration with MLIR contexts.
//!
//! # Requirements
//!
//! This crate requires an LLVM/MLIR installation with `mlir-tblgen`. The build
//! process locates LLVM in the following order:
//!
//! 1. `llvm-config` in PATH (recommended)
//! 2. `LLVM_PREFIX` environment variable
//!
//! If `llvm-config` is available, version-specific environment variables like
//! `MLIR_SYS_210_PREFIX` (for LLVM 21) are also checked as overrides.
//!
//! # Usage
//!
//! ## Single File
//!
//! In your `build.rs`:
//!
//! ```rust,ignore
//! use melior_build::DialectBuilder;
//!
//! fn main() {
//!     DialectBuilder::new("my_dialect")
//!         .td_file("src/dialect/MyDialect.td")
//!         .build()
//!         .expect("Failed to build dialect");
//! }
//! ```
//!
//! ## Multiple Files
//!
//! For split TableGen files (like many MLIR projects), provide each file
//! separately. The builder auto-detects what each file contains and runs
//! only the relevant generators:
//!
//! ```rust,ignore
//! use melior_build::DialectBuilder;
//!
//! fn main() {
//!     DialectBuilder::new("bril")
//!         .td_file("src/dialect/bril/BrilDialect.td")  // dialect definition
//!         .td_file("src/dialect/bril/BrilTypes.td")    // type definitions
//!         .td_file("src/dialect/bril/BrilOps.td")      // operation definitions
//!         .include_dir("src/dialect")                   // for include resolution
//!         .cpp_namespace("mlir::bril")
//!         .build()
//!         .expect("Failed to build dialect");
//! }
//! ```
//!
//! ## Additional C++ Sources
//!
//! If your dialect requires additional C++ implementation files (e.g., for
//! custom verifiers, canonicalizers, or builders), you can include them:
//!
//! ```rust,ignore
//! use melior_build::DialectBuilder;
//!
//! fn main() {
//!     DialectBuilder::new("my_dialect")
//!         .td_file("src/dialect/MyDialect.td")
//!         .cpp_file("src/dialect/MyDialectImpl.cpp")
//!         .cpp_files(["src/dialect/Canonicalize.cpp", "src/dialect/Verifiers.cpp"])
//!         .build()
//!         .expect("Failed to build dialect");
//! }
//! ```
//!
//! In your `lib.rs`:
//!
//! ```rust,ignore
//! // Include the generated registration code
//! include!(concat!(env!("OUT_DIR"), "/my_dialect_register.rs"));
//!
//! // Now you can use my_dialect::register(&context)
//! ```
//!
//! # melior-build vs melior::dialect! macro
//!
//! | Feature | `melior-build` | `melior::dialect!` macro |
//! |---------|----------------|--------------------------|
//! | **Purpose** | C++ dialect registration | Rust operation wrappers |
//! | **TableGen parser** | Official `mlir-tblgen` | Rust `tblgen` crate |
//! | **Multi-file support** | Yes (auto-detection) | Single file recommended |
//! | **Used in** | `build.rs` | `lib.rs` |

pub mod cpp_gen;
mod error;
pub mod rust_gen;
pub mod tblgen;

pub use error::Error;

use std::path::{Path, PathBuf};

/// Convert a dialect name to CamelCase class name.
/// e.g., "math_ext" -> "MathExt", "my_dialect" -> "MyDialect"
pub(crate) fn to_class_name(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

/// Builder for compiling and registering a custom MLIR dialect.
///
/// This builder handles:
/// 1. Running `mlir-tblgen` to generate C++ dialect code
/// 2. Generating C API registration wrappers
/// 3. Compiling the C++ into a static library
/// 4. Generating Rust FFI bindings
///
/// The builder auto-detects what each TableGen file contains (dialect
/// definitions, operations, types, attributes, enums, FunctionOpInterface) and
/// runs only the relevant generators.
#[derive(Debug, Clone)]
pub struct DialectBuilder {
    /// The dialect name (e.g., "toy")
    name: String,
    /// The C++ namespace for the dialect (e.g., "mlir::toy")
    cpp_namespace: Option<String>,
    /// TableGen files to process
    td_files: Vec<PathBuf>,
    /// Include directories for TableGen
    include_dirs: Vec<PathBuf>,
    /// Additional C++ source files to compile
    cpp_files: Vec<PathBuf>,
    /// Output directory (defaults to OUT_DIR)
    output_dir: Option<PathBuf>,
}

impl DialectBuilder {
    /// Create a new dialect builder with the given dialect name.
    ///
    /// The name should match the dialect name in your TableGen definition.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cpp_namespace: None,
            td_files: Vec::new(),
            include_dirs: Vec::new(),
            cpp_files: Vec::new(),
            output_dir: None,
        }
    }

    /// Extracts subdirectory path from cpp_namespace.
    ///
    /// Returns:
    /// - `Ok(Some("bril"))` for `"mlir::bril"`
    /// - `Ok(None)` if cpp_namespace is not set
    /// - `Err` for single-level namespaces (must use `mlir::X` pattern)
    /// - `Err` for namespaces with 3+ levels
    /// - `Err` for malformed namespaces (leading/trailing `::`)
    fn namespace_subdir(&self) -> Result<Option<String>, Error> {
        match &self.cpp_namespace {
            None => Ok(None),
            Some(ns) => {
                let trimmed = ns.trim();
                if trimmed.is_empty() {
                    return Ok(None);
                }

                // Reject leading or trailing ::
                if trimmed.starts_with("::") || trimmed.ends_with("::") {
                    return Err(Error::InvalidNamespace(format!(
                        "cpp_namespace '{}' has invalid leading or trailing '::'.",
                        ns
                    )));
                }

                let parts: Vec<&str> = trimmed.split("::").collect();
                match parts.as_slice() {
                    ["mlir", dialect] => Ok(Some((*dialect).to_string())),
                    [single] => Err(Error::InvalidNamespace(format!(
                        "cpp_namespace '{}' must use the 'mlir::namespace' pattern. \
                         Did you mean 'mlir::{}'?",
                        ns, single
                    ))),
                    _ => Err(Error::InvalidNamespace(format!(
                        "cpp_namespace '{}' has more than 2 levels. \
                         Only 'mlir::namespace' pattern is supported.",
                        ns
                    ))),
                }
            }
        }
    }

    /// Set the C++ namespace for the dialect.
    ///
    /// The namespace must follow the `mlir::name` pattern (e.g., `mlir::bril`).
    /// This determines both the C++ namespace wrapping and the subdirectory
    /// for generated `.inc` files (e.g., `inc/bril/BrilOps.h.inc`).
    ///
    /// If not set, defaults to `mlir::{name}` and files are placed directly
    /// in the `inc/` directory without a subdirectory.
    ///
    /// # Errors
    ///
    /// The build will fail if the namespace:
    /// - Has only one level (e.g., `"bril"` instead of `"mlir::bril"`)
    /// - Has more than two levels (e.g., `"mlir::foo::bar"`)
    /// - Has leading or trailing `::` (e.g., `"mlir::bril::"`)
    pub fn cpp_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.cpp_namespace = Some(namespace.into());
        self
    }

    /// Add a TableGen file to process.
    pub fn td_file(mut self, path: impl AsRef<Path>) -> Self {
        self.td_files.push(path.as_ref().to_path_buf());
        self
    }

    /// Add multiple TableGen files to process.
    pub fn td_files<P: AsRef<Path>>(mut self, paths: impl IntoIterator<Item = P>) -> Self {
        self.td_files
            .extend(paths.into_iter().map(|p| p.as_ref().to_path_buf()));
        self
    }

    /// Add an include directory for TableGen processing.
    pub fn include_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.include_dirs.push(path.as_ref().to_path_buf());
        self
    }

    /// Add multiple include directories for TableGen processing.
    pub fn include_dirs<P: AsRef<Path>>(mut self, paths: impl IntoIterator<Item = P>) -> Self {
        self.include_dirs
            .extend(paths.into_iter().map(|p| p.as_ref().to_path_buf()));
        self
    }

    /// Add an additional C++ source file to compile.
    ///
    /// Use this for custom verifiers, canonicalizers, builders, or other
    /// C++ implementations required by your TableGen definitions.
    pub fn cpp_file(mut self, path: impl AsRef<Path>) -> Self {
        self.cpp_files.push(path.as_ref().to_path_buf());
        self
    }

    /// Add multiple additional C++ source files to compile.
    pub fn cpp_files<P: AsRef<Path>>(mut self, paths: impl IntoIterator<Item = P>) -> Self {
        self.cpp_files
            .extend(paths.into_iter().map(|p| p.as_ref().to_path_buf()));
        self
    }

    /// Set the output directory for generated files.
    ///
    /// If not set, defaults to the `OUT_DIR` environment variable.
    pub fn output_dir(mut self, path: impl AsRef<Path>) -> Self {
        self.output_dir = Some(path.as_ref().to_path_buf());
        self
    }

    /// Build the dialect registration code.
    ///
    /// This will:
    /// 1. Run `mlir-tblgen` to generate C++ `.inc` files
    /// 2. Generate a C++ wrapper file with registration code
    /// 3. Compile the C++ code into a static library
    /// 4. Generate Rust FFI bindings
    ///
    /// The generated Rust file should be included in your crate:
    /// ```rust,ignore
    /// include!(concat!(env!("OUT_DIR"), "/{name}_register.rs"));
    /// ```
    pub fn build(self) -> Result<(), Error> {
        let output_dir = self.get_output_dir()?;
        let llvm_prefix = self.get_llvm_prefix()?;
        let cpp_namespace = self
            .cpp_namespace
            .clone()
            .unwrap_or_else(|| format!("mlir::{}", self.name));

        std::fs::create_dir_all(&output_dir)?;

        let tblgen_runner = tblgen::TblgenRunner::new(&llvm_prefix)?;

        // Create base inc/ directory
        let inc_base = output_dir.join("inc");
        std::fs::create_dir_all(&inc_base)?;

        // Get namespace-based subdirectory (e.g., "mlir::bril" -> "bril")
        let inc_subdir = self.namespace_subdir()?;

        // Create the actual output directory for .inc files
        let inc_dir = match &inc_subdir {
            Some(subdir) => {
                let dir = inc_base.join(subdir);
                std::fs::create_dir_all(&dir)?;
                dir
            }
            None => inc_base.clone(),
        };

        let mut detected_types = false;
        let mut detected_attrs = false;
        let mut detected_enums = false;
        let mut detected_function_interface = false;

        for td_file in &self.td_files {
            let contents = tblgen::detect_td_contents(td_file)?;

            detected_types |= contents.has_types;
            detected_attrs |= contents.has_attrs;
            detected_enums |= contents.has_enums;
            detected_function_interface |= contents.has_function_interface;

            tblgen_runner.generate_for_file(
                td_file,
                &self.include_dirs,
                &inc_dir,
                &self.name,
                &contents,
            )?;
        }

        let gen_options = tblgen::GenerationOptions {
            generate_types: detected_types,
            generate_attributes: detected_attrs,
            generate_enums: detected_enums,
            use_function_interface: detected_function_interface,
        };

        let cpp_file = output_dir.join(format!("{}_capi.cpp", self.name));
        cpp_gen::generate_cpp_registration(
            &self.name,
            &cpp_namespace,
            &gen_options,
            inc_subdir.as_deref(),
            &cpp_file,
        )?;

        Self::compile_cpp(
            &self.name,
            &cpp_file,
            &self.cpp_files,
            &self.include_dirs,
            &inc_base, // Use base inc/ dir so includes like "bril/BrilOps.h.inc" resolve
            &llvm_prefix,
        )?;

        let rust_file = output_dir.join(format!("{}_register.rs", self.name));
        rust_gen::generate_rust_ffi(&self.name, &rust_file)?;

        for td_file in &self.td_files {
            println!("cargo:rerun-if-changed={}", td_file.display());
        }

        for cpp_file in &self.cpp_files {
            println!("cargo:rerun-if-changed={}", cpp_file.display());
        }

        Ok(())
    }

    fn get_output_dir(&self) -> Result<PathBuf, Error> {
        if let Some(ref dir) = self.output_dir {
            Ok(dir.clone())
        } else {
            std::env::var("OUT_DIR")
                .map(PathBuf::from)
                .map_err(|_| Error::MissingOutDir)
        }
    }

    fn get_llvm_prefix(&self) -> Result<PathBuf, Error> {
        // Try llvm-config first (most reliable when available)
        if let Some(prefix) = Self::llvm_config("--prefix") {
            // Also check version-specific env var in case user wants to override
            if let Some(version) = Self::llvm_config("--version")
                && let Some(major) = version
                    .split('.')
                    .next()
                    .and_then(|v| v.parse::<u32>().ok())
            {
                let var = format!("MLIR_SYS_{}0_PREFIX", major);
                if let Ok(p) = std::env::var(&var) {
                    return Ok(PathBuf::from(p));
                }
            }
            return Ok(PathBuf::from(prefix));
        }

        // Fallback to generic env var
        if let Ok(prefix) = std::env::var("LLVM_PREFIX") {
            return Ok(PathBuf::from(prefix));
        }

        Err(Error::LlvmNotFound)
    }

    fn llvm_config(arg: &str) -> Option<String> {
        let output = std::process::Command::new("llvm-config")
            .arg(arg)
            .output()
            .ok()?;
        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            None
        }
    }

    fn compile_cpp(
        name: &str,
        cpp_file: &Path,
        additional_cpp_files: &[PathBuf],
        include_dirs: &[PathBuf],
        inc_dir: &Path,
        llvm_prefix: &Path,
    ) -> Result<(), Error> {
        let llvm_include = llvm_prefix.join("include");

        let mut build = cc::Build::new();
        build
            .file(cpp_file)
            .cpp(true)
            .std("c++17")
            .include(inc_dir)
            .define("MLIR_CAPI_BUILDING_LIBRARY", "1")
            .flag_if_supported("-fno-rtti")
            .flag_if_supported("-fno-exceptions")
            // Suppress warnings from LLVM/MLIR headers and generated code
            .flag_if_supported(format!("-isystem{}", llvm_include.display()))
            .flag_if_supported(format!("-isystem{}", inc_dir.display()))
            .flag_if_supported("-Wno-unused-parameter");

        // Add user-specified include directories
        for dir in include_dirs {
            build.include(dir);
        }

        // Add additional C++ source files
        for file in additional_cpp_files {
            build.file(file);
        }

        #[cfg(target_os = "macos")]
        build.cpp_link_stdlib("c++");
        #[cfg(target_os = "linux")]
        build.cpp_link_stdlib("stdc++");

        build.compile(&format!("{}_dialect", name));

        let lib_dir = llvm_prefix.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=MLIRIR");
        println!("cargo:rustc-link-lib=MLIRSupport");
        println!("cargo:rustc-link-lib=MLIRCAPIIR");

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_subdir_mlir_prefix() {
        let builder = DialectBuilder::new("test").cpp_namespace("mlir::bril");
        assert_eq!(
            builder.namespace_subdir().unwrap(),
            Some("bril".to_string())
        );
    }

    #[test]
    fn test_namespace_subdir_single_level_errors() {
        // Single-level namespace should error - must use mlir::X pattern
        let builder = DialectBuilder::new("test").cpp_namespace("bril");
        let err = builder.namespace_subdir().unwrap_err();
        assert!(
            err.to_string()
                .contains("must use the 'mlir::namespace' pattern")
        );
        assert!(err.to_string().contains("Did you mean 'mlir::bril'"));
    }

    #[test]
    fn test_namespace_subdir_too_deep() {
        let builder = DialectBuilder::new("test").cpp_namespace("mlir::foo::bar");
        let err = builder.namespace_subdir().unwrap_err();
        assert!(err.to_string().contains("has more than 2 levels"));
    }

    #[test]
    fn test_namespace_subdir_none() {
        let builder = DialectBuilder::new("test");
        assert_eq!(builder.namespace_subdir().unwrap(), None);
    }

    #[test]
    fn test_namespace_subdir_empty_string() {
        let builder = DialectBuilder::new("test").cpp_namespace("");
        assert_eq!(builder.namespace_subdir().unwrap(), None);
    }

    #[test]
    fn test_namespace_subdir_whitespace() {
        let builder = DialectBuilder::new("test").cpp_namespace("  mlir::bril  ");
        assert_eq!(
            builder.namespace_subdir().unwrap(),
            Some("bril".to_string())
        );
    }

    #[test]
    fn test_namespace_subdir_only_mlir() {
        // "mlir" alone should error (single-level)
        let builder = DialectBuilder::new("test").cpp_namespace("mlir");
        let err = builder.namespace_subdir().unwrap_err();
        assert!(
            err.to_string()
                .contains("must use the 'mlir::namespace' pattern")
        );
    }

    #[test]
    fn test_namespace_subdir_trailing_colons() {
        let builder = DialectBuilder::new("test").cpp_namespace("mlir::bril::");
        let err = builder.namespace_subdir().unwrap_err();
        assert!(err.to_string().contains("invalid leading or trailing '::'"));
    }

    #[test]
    fn test_namespace_subdir_leading_colons() {
        let builder = DialectBuilder::new("test").cpp_namespace("::mlir::bril");
        let err = builder.namespace_subdir().unwrap_err();
        assert!(err.to_string().contains("invalid leading or trailing '::'"));
    }
}
