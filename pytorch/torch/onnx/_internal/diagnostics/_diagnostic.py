class TorchScriptOnnxExportDiagnostic(infra.Diagnostic):
    """Base class for all export diagnostics.

    This class is used to represent all export diagnostics. It is a subclass of
    infra.Diagnostic, and adds additional methods to add more information to the
    diagnostic.
    """

    python_call_stack: Optional[infra.Stack] = None
    cpp_call_stack: Optional[infra.Stack] = None

    def __init__(
        self,
        *args,
        frames_to_skip: int = 1,
        cpp_stack: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.python_call_stack = self.record_python_call_stack(
            frames_to_skip=frames_to_skip
        )
        if cpp_stack:
            self.cpp_call_stack = self.record_cpp_call_stack(
                frames_to_skip=frames_to_skip
            )

    def record_cpp_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current C++ call stack in the diagnostic."""
        # NOTE: Cannot use `@_beartype.beartype`. It somehow erases the cpp stack frame info.
        # No need to skip this function because python frame is not recorded
        # in cpp call stack.
        stack = _cpp_call_stack(frames_to_skip=frames_to_skip)
        stack.message = "C++ call stack"
        self.with_stack(stack)
        return stack

