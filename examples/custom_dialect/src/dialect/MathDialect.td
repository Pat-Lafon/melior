// A minimal custom dialect for testing melior-build.
//
// This dialect defines a simple "math_ext" dialect with basic arithmetic
// operations that demonstrate custom dialect registration.

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def MathExt_Dialect : Dialect {
    let name = "math_ext";
    let summary = "A simple extended math dialect for testing.";
    let description = [{
        This dialect provides extended mathematical operations as an example
        of how to define and register custom dialects with melior-build.
    }];
    let cppNamespace = "::mlir::math_ext";
}

//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

class MathExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<MathExt_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def MathExt_AddOp : MathExt_Op<"add", [Pure]> {
    let summary = "Extended addition operation";
    let description = [{
        Performs addition of two integer values.

        Example:
        ```mlir
        %result = math_ext.add %a, %b : i32
        ```
    }];

    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger:$result);
}

def MathExt_MulOp : MathExt_Op<"mul", [Pure]> {
    let summary = "Extended multiplication operation";
    let description = [{
        Performs multiplication of two integer values.
    }];

    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger:$result);
}

def MathExt_ConstOp : MathExt_Op<"const", [Pure]> {
    let summary = "Integer constant operation";
    let description = [{
        Produces a constant integer value.

        Example:
        ```mlir
        %c42 = math_ext.const 42 : i32
        ```
    }];

    let arguments = (ins I64Attr:$value);
    let results = (outs AnyInteger:$result);
}
