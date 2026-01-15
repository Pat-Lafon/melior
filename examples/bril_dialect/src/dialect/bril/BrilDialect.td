// Bril dialect definition for MLIR.
//
// Based on the Bril intermediate language: https://capra.cs.cornell.edu/bril/
// This dialect closely follows the original brilir MLIR implementation.

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect Definition
//===----------------------------------------------------------------------===//

def Bril_Dialect : Dialect {
    let name = "bril";
    let summary = "An MLIR dialect for the Bril intermediate language.";
    let description = [{
        Bril (Big Red Intermediate Language) is a simple intermediate
        representation for teaching compiler concepts. This dialect
        provides MLIR operations for Bril programs.

        See: https://capra.cs.cornell.edu/bril/
    }];
    let cppNamespace = "::mlir::bril";
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base Operation Definition
//===----------------------------------------------------------------------===//

class Bril_Op<string mnemonic, list<Trait> traits = []> :
    Op<Bril_Dialect, mnemonic, traits>;
