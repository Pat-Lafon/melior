// Bril type definitions for MLIR.

include "mlir/IR/AttrTypeBase.td"
include "bril/BrilDialect.td"

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

class Bril_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Bril_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Bril_PtrType : Bril_Type<"Ptr", "ptr"> {
    let summary = "Bril pointer type";
    let description = [{
        A pointer type that points to a value of the given element type.
        This is part of the Bril memory extension.

        Example:
        ```mlir
        %ptr : !bril.ptr<i64>
        ```
    }];
    let parameters = (ins "::mlir::Type":$pointeeType);
    let assemblyFormat = "`<` $pointeeType `>`";
}

//===----------------------------------------------------------------------===//
// Type Constraint
//===----------------------------------------------------------------------===//

def ValidBrilType : AnyTypeOf<[I64, I1, Bril_PtrType]>;
