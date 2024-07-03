def parse(message):
    """A simple parser that parses the report of cuda-memcheck. This parser is meant to be simple
    and it only split the report into separate errors and a summary. Where each error is further
    splitted into error message and backtrace. No further details are parsed.

    A report contains multiple errors and a summary on how many errors are detected. It looks like:

    ========= CUDA-MEMCHECK
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaPointerGetAttributes.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaPointerGetAttributes + 0x1a9) [0x428b9]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b778a9]
    =========     .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     .....
    =========
    ========= .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     .....
    =========     Host Frame:python (_PyEval_EvalFrameDefault + 0x6a0) [0x1d0ad0]
    =========     Host Frame:python (_PyEval_EvalCodeWithName + 0xbb9) [0x116db9]
    =========
    ========= ERROR SUMMARY: 4 errors
    """
    errors = []
    HEAD = "========="
    headlen = len(HEAD)
    started = False
    in_message = False
    message_lines = []
    lines = message.splitlines()
    for l in lines:
        if l == HEAD + " CUDA-MEMCHECK":
            started = True
            continue
        if not started or not l.startswith(HEAD):
            continue
        l = l[headlen + 1 :]
        if l.startswith("ERROR SUMMARY:"):
            return Report(l, errors)
        if not in_message:
            in_message = True
            message_lines = [l]
        elif l == "":
            errors.append(Error(message_lines))
            in_message = False
        else:
            message_lines.append(l)
    raise ParseError("No error summary found")def parse(message):
    """A simple parser that parses the report of cuda-memcheck. This parser is meant to be simple
    and it only split the report into separate errors and a summary. Where each error is further
    splitted into error message and backtrace. No further details are parsed.

    A report contains multiple errors and a summary on how many errors are detected. It looks like:

    ========= CUDA-MEMCHECK
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaPointerGetAttributes.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaPointerGetAttributes + 0x1a9) [0x428b9]
    =========     Host Frame:/home/xgao/anaconda3/lib/python3.7/site-packages/torch/lib/libtorch.so [0x5b778a9]
    =========     .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     Host Frame:/usr/local/cuda/lib64/libcudart.so.10.1 (cudaGetLastError + 0x163) [0x4c493]
    =========     .....
    =========
    ========= .....
    =========
    ========= Program hit cudaErrorInvalidValue (error 1) due to "invalid argument" on CUDA API call to cudaGetLastError.
    =========     Saved host backtrace up to driver entry point at error
    =========     Host Frame:/usr/lib/x86_64-linux-gnu/libcuda.so.1 [0x38c7b3]
    =========     .....
    =========     Host Frame:python (_PyEval_EvalFrameDefault + 0x6a0) [0x1d0ad0]
    =========     Host Frame:python (_PyEval_EvalCodeWithName + 0xbb9) [0x116db9]
    =========
    ========= ERROR SUMMARY: 4 errors
    """
    errors = []
    HEAD = "========="
    headlen = len(HEAD)
    started = False
    in_message = False
    message_lines = []
    lines = message.splitlines()
    for l in lines:
        if l == HEAD + " CUDA-MEMCHECK":
            started = True
            continue
        if not started or not l.startswith(HEAD):
            continue
        l = l[headlen + 1 :]
        if l.startswith("ERROR SUMMARY:"):
            return Report(l, errors)
        if not in_message:
            in_message = True
            message_lines = [l]
        elif l == "":
            errors.append(Error(message_lines))
            in_message = False
        else:
            message_lines.append(l)
    raise ParseError("No error summary found")