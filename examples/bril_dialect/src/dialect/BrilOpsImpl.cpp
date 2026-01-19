// Custom verifier implementations for Bril dialect operations.
//
// This file provides verifiers for operations that require custom C++
// validation logic, such as checking pointer/pointee type consistency.

// MLIR core headers
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"

// Interface headers required by the generated operations
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// For TypeSwitch
#include "llvm/ADT/TypeSwitch.h"

// Include generated dialect declaration
#include "BrilDialect.h.inc"

// Include generated type definitions (needed for PtrType)
#define GET_TYPEDEF_CLASSES
#include "BrilTypes.h.inc"

// Include generated operation declarations with full class definitions
#define GET_OP_CLASSES
#include "BrilOps.h.inc"

namespace mlir {
namespace bril {

//===----------------------------------------------------------------------===//
// LoadOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  auto ptrType = dyn_cast<PtrType>(getPtr().getType());
  if (!ptrType)
    return emitOpError("expected 'ptr' type for 'ptr' operand");

  if (getResult().getType() != ptrType.getPointeeType())
    return emitOpError("result type must match pointee type of pointer");

  return success();
}

//===----------------------------------------------------------------------===//
// StoreOp Verifier
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::verify() {
  auto ptrType = dyn_cast<PtrType>(getPtr().getType());
  if (!ptrType)
    return emitOpError("expected 'ptr' type for 'ptr' operand");

  if (getValue().getType() != ptrType.getPointeeType())
    return emitOpError("value type must match pointee type of pointer");

  return success();
}

} // namespace bril
} // namespace mlir
