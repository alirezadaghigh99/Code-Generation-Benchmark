class ComputeNativeFunctionStub:
    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str | None:
        if Variant.function not in f.variants:
            return None

        sig = DispatcherSignature.from_schema(
            f.func, prefix=f"wrapper_CPU_{f.func.name.overload_name}_", symint=False
        )
        assert sig is not None
        if len(f.func.returns) == 0:
            ret_name = ""
        elif len(f.func.returns) == 1:
            if f.func.arguments.out:
                ret_name = f.func.arguments.out[0].name
            else:
                ret_name = next(
                    (
                        a.name
                        for a in f.func.arguments.flat_non_out
                        if a.type == f.func.returns[0].type
                    ),
                    "",
                )
            if not ret_name:
                # if return type is tensor
                if f.func.returns[0].type == BaseType(BaseTy.Tensor):
                    # Returns an empty tensor
                    ret_name = "at::Tensor()"
                else:
                    raise Exception(  # noqa: TRY002
                        f"Can't handle this return type {f.func}"
                    )  # noqa: TRY002
        elif len(f.func.arguments.out) == len(f.func.returns):
            # Returns a tuple of out arguments
            tensor_type = "at::Tensor &"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join([r.name for r in f.func.arguments.out])}
            )"""
        else:
            assert all(
                a.type == BaseType(BaseTy.Tensor) for a in f.func.returns
            ), f"Only support tensor returns but got {f.func.returns}"
            # Returns a tuple of empty tensors
            tensor_type = "at::Tensor"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join(["at::Tensor()" for _ in f.func.returns])}
            )"""
        ret_str = f"return {ret_name};" if len(f.func.returns) > 0 else ""
        return f"""
{sig.defn()} {{
    {ret_str}
}}
    """