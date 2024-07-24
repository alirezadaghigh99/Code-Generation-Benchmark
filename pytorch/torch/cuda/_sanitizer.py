class TensorInfo:
    r"""Stores information about a single tensor and recent accesses to it.

    Args:
        allocation_stack_trace: the stack summary object captured during tensor
            allocation. Can be ``None`` if the allocation wasn't caught by CSAN.
        reads: list of read accesses to the tensor that were performed since
            the last write.
        write: the last write access to the tensor.
    """

    allocation_stack_trace: Optional[traceback.StackSummary]
    reads: List[Access] = field(default_factory=list)
    write: Optional[Access] = None

