def issubdtype(arg1, arg2):
    # cf https://github.com/numpy/numpy/blob/v1.24.0/numpy/core/numerictypes.py#L356-L420

    # We also accept strings even if NumPy doesn't as dtypes are serialized as their
    # string representation in dynamo's graph
    def str_to_abstract(t):
        if isinstance(t, str) and t in _abstract_dtypes:
            return globals()[t]
        return t

    arg1 = str_to_abstract(arg1)
    arg2 = str_to_abstract(arg2)

    if not issubclass_(arg1, generic):
        arg1 = dtype(arg1).type
    if not issubclass_(arg2, generic):
        arg2 = dtype(arg2).type
    return issubclass(arg1, arg2)

class int8(signedinteger):
    name = "int8"
    typecode = "b"
    torch_dtype = torch.int8

class uint64(signedinteger):
    name = "uint64"
    typecode = "L"
    torch_dtype = torch.uint64

class float64(floating):
    name = "float64"
    typecode = "d"
    torch_dtype = torch.float64

class int16(signedinteger):
    name = "int16"
    typecode = "h"
    torch_dtype = torch.int16

class uint8(unsignedinteger):
    name = "uint8"
    typecode = "B"
    torch_dtype = torch.uint8

class complex64(complexfloating):
    name = "complex64"
    typecode = "F"
    torch_dtype = torch.complex64

