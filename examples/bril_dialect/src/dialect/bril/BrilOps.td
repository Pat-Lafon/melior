// Bril operation definitions for MLIR.

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"

// BrilTypes.td already includes BrilDialect.td, so we only need to include BrilTypes.td
include "bril/BrilTypes.td"

//===----------------------------------------------------------------------===//
// Arithmetic Operations
//===----------------------------------------------------------------------===//

def Bril_ConstantOp : Bril_Op<"const", [Pure]> {
    let summary = "Defines a constant value.";
    let description = [{
        The `bril.const` operation defines a constant integer or boolean value.

        Example:
        ```mlir
        %c42 = bril.const 42 : i64
        %true = bril.const true : i1
        ```
    }];

    let arguments = (ins AnyAttrOf<[I64Attr, I1Attr]>:$value);
    let results = (outs AnyTypeOf<[I64, I1]>:$result);
}

def Bril_AddOp : Bril_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "Adds two integers.";
    let description = [{
        The `bril.add` operation takes two integer operands and returns their sum.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}

def Bril_SubOp : Bril_Op<"sub", [Pure, SameOperandsAndResultType]> {
    let summary = "Subtracts two integers.";
    let description = [{
        The `bril.sub` operation takes two integer operands and returns the
        result of subtracting the second from the first.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}

def Bril_MulOp : Bril_Op<"mul", [Pure, SameOperandsAndResultType]> {
    let summary = "Multiplies two integers.";
    let description = [{
        The `bril.mul` operation takes two integer operands and returns their product.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}

def Bril_DivOp : Bril_Op<"div", [SameOperandsAndResultType]> {
    let summary = "Divides two integers.";
    let description = [{
        The `bril.div` operation takes two integer operands and returns the
        result of dividing the first by the second.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def Bril_EqOp : Bril_Op<"eq", [Pure]> {
    let summary = "Compares two integers for equality.";
    let description = [{
        The `bril.eq` operation takes two integer operands and returns true
        if they are equal, otherwise returns false.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I1);
}

def Bril_LtOp : Bril_Op<"lt", [Pure]> {
    let summary = "Compares two integers for less-than.";
    let description = [{
        The `bril.lt` operation takes two integer operands and returns true
        if the first is less than the second, otherwise returns false.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I1);
}

def Bril_GtOp : Bril_Op<"gt", [Pure]> {
    let summary = "Compares two integers for greater-than.";
    let description = [{
        The `bril.gt` operation takes two integer operands and returns true
        if the first is greater than the second, otherwise returns false.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I1);
}

def Bril_LeOp : Bril_Op<"le", [Pure]> {
    let summary = "Compares two integers for less-than-or-equal.";
    let description = [{
        The `bril.le` operation takes two integer operands and returns true
        if the first is less than or equal to the second, otherwise returns false.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I1);
}

def Bril_GeOp : Bril_Op<"ge", [Pure]> {
    let summary = "Compares two integers for greater-than-or-equal.";
    let description = [{
        The `bril.ge` operation takes two integer operands and returns true
        if the first is greater than or equal to the second, otherwise returns false.
    }];
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I1);
}

//===----------------------------------------------------------------------===//
// Logical Operations
//===----------------------------------------------------------------------===//

def Bril_NotOp : Bril_Op<"not", [Pure]> {
    let summary = "Logical NOT operation.";
    let description = [{
        The `bril.not` operation takes a boolean operand and returns its
        logical negation.
    }];
    let arguments = (ins I1:$operand);
    let results = (outs I1);
}

def Bril_AndOp : Bril_Op<"and", [Pure]> {
    let summary = "Logical AND operation.";
    let description = [{
        The `bril.and` operation takes two boolean operands and returns true
        if both are true, otherwise returns false.
    }];
    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1);
}

def Bril_OrOp : Bril_Op<"or", [Pure]> {
    let summary = "Logical OR operation.";
    let description = [{
        The `bril.or` operation takes two boolean operands and returns true
        if at least one is true, otherwise returns false.
    }];
    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1);
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def Bril_IdOp : Bril_Op<"id", [Pure]> {
    let summary = "Identity operation.";
    let description = [{
        The `bril.id` operation returns its input operand unchanged.
        Useful for representing variable assignments in SSA form.
    }];
    let arguments = (ins ValidBrilType:$input);
    let results = (outs ValidBrilType);
}

def Bril_UndefOp : Bril_Op<"undef"> {
    let summary = "Defines an undefined value.";
    let description = [{
        The `bril.undef` operation produces an undefined value.
    }];
    let results = (outs ValidBrilType);
}

def Bril_NopOp : Bril_Op<"nop"> {
    let summary = "No operation.";
    let description = [{
        The `bril.nop` operation performs no action.
    }];
}

//===----------------------------------------------------------------------===//
// I/O Operations
//===----------------------------------------------------------------------===//

def Bril_PrintOp : Bril_Op<"print"> {
    let summary = "Prints values.";
    let description = [{
        The `bril.print` operation outputs values to standard output.
    }];
    let arguments = (ins Variadic<AnyTypeOf<[I64, I1]>>:$values);
}

//===----------------------------------------------------------------------===//
// Memory Operations
//===----------------------------------------------------------------------===//

def Bril_AllocOp : Bril_Op<"alloc"> {
    let summary = "Allocates memory.";
    let description = [{
        The `bril.alloc` operation allocates memory for a given type.

        Example:
        ```mlir
        %ptr = bril.alloc %size : !bril.ptr<i64>
        ```
    }];
    let arguments = (ins I64:$size);
    let results = (outs Bril_PtrType);
}

def Bril_FreeOp : Bril_Op<"free"> {
    let summary = "Frees allocated memory.";
    let description = [{
        The `bril.free` operation deallocates memory pointed to by a pointer.
    }];
    let arguments = (ins Bril_PtrType:$ptr);
}

def Bril_LoadOp : Bril_Op<"load"> {
    let summary = "Loads a value from memory.";
    let description = [{
        The `bril.load` operation loads a value from the memory location
        pointed to by a pointer.

        The result type must match the pointer's pointee type.
    }];
    let arguments = (ins Bril_PtrType:$ptr);
    let results = (outs ValidBrilType);
    let hasVerifier = 1;
}

def Bril_StoreOp : Bril_Op<"store"> {
    let summary = "Stores a value to memory.";
    let description = [{
        The `bril.store` operation stores a value to the memory location
        pointed to by a pointer.

        The value type must match the pointer's pointee type.
    }];
    let arguments = (ins Bril_PtrType:$ptr, ValidBrilType:$value);
    let hasVerifier = 1;
}

def Bril_PtrAddOp : Bril_Op<"ptr_add", [Pure]> {
    let summary = "Adds an integer offset to a pointer.";
    let description = [{
        The `bril.ptr_add` operation takes a pointer and an integer offset,
        returning a new pointer offset by the given amount.
    }];
    let arguments = (ins Bril_PtrType:$ptr, I64:$offset);
    let results = (outs Bril_PtrType);
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

def Bril_CallOp : Bril_Op<"call"> {
    let summary = "Calls a function.";
    let description = [{
        The `bril.call` operation invokes a function with the given arguments
        and returns the result.
    }];
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<ValidBrilType>:$inputs,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );
    let results = (outs Optional<ValidBrilType>);

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];
}

def Bril_JmpOp : Bril_Op<"jmp", [Pure, Terminator]> {
    let summary = "Unconditional jump to a label.";
    let description = [{
        The `bril.jmp` operation transfers control unconditionally to the
        specified label.
    }];
    let arguments = (ins Variadic<ValidBrilType>:$args);
    let successors = (successor AnySuccessor:$target);

    let assemblyFormat = [{
        $target (`(` $args^ `:` type($args) `)`)? attr-dict
    }];
}

def Bril_BrOp : Bril_Op<"br", [Pure, Terminator, AttrSizedOperandSegments]> {
    let summary = "Conditional branch to labels.";
    let description = [{
        The `bril.br` operation transfers control to one of two specified
        labels based on a boolean condition.
    }];
    let arguments = (ins I1:$condition, Variadic<ValidBrilType>:$true_args, Variadic<ValidBrilType>:$false_args);
    let successors = (successor AnySuccessor:$true_target, AnySuccessor:$false_target);

    let assemblyFormat = [{
        $condition `,`
        $true_target (`(` $true_args^ `:` type($true_args) `)`)? `,`
        $false_target (`(` $false_args^ `:` type($false_args) `)`)?
        attr-dict
    }];
}

def Bril_FuncOp : Bril_Op<"func", [FunctionOpInterface, IsolatedFromAbove]> {
    let summary = "Defines a function.";
    let description = [{
        The `bril.func` operation defines a function with a name, arguments,
        return type, and body.
    }];
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        //===------------------------------------------------------------------===//
        // FunctionOpInterface Methods
        //===------------------------------------------------------------------===//

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

        /// Returns the region on the function operation that is callable.
        Region *getCallableRegion() { return &getBody(); }
    }];

    let builders = [OpBuilder<(ins "StringRef":$name, "FunctionType":$type, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>];
    let skipDefaultBuilders = 1;
}

def Bril_RetOp : Bril_Op<"ret", [Pure, Terminator]> {
    let summary = "Return from a function.";
    let description = [{
        The `bril.ret` operation returns control from a function, optionally
        returning a value.
    }];
    let arguments = (ins Optional<ValidBrilType>:$return_value);
}
