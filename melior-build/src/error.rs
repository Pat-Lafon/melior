//! Error types for melior-build.

use std::path::PathBuf;

/// Errors that can occur during dialect building.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The OUT_DIR environment variable is not set.
    #[error("OUT_DIR environment variable not set. This crate must be used from a build script.")]
    MissingOutDir,

    /// LLVM/MLIR installation could not be found.
    #[error(
        "Could not find LLVM/MLIR installation. Ensure llvm-config is in PATH, or set LLVM_PREFIX."
    )]
    LlvmNotFound,

    /// mlir-tblgen binary could not be found.
    #[error("Could not find mlir-tblgen binary at {0}")]
    TblgenNotFound(PathBuf),

    /// mlir-tblgen execution failed.
    #[error("mlir-tblgen failed: {0}")]
    TblgenFailed(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// UTF-8 conversion error.
    #[error("UTF-8 conversion error: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}
