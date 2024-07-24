class AvoidOOM:
    """Try to convert inputs to FP16 and CPU if got a PyTorch's CUDA Out of
    Memory error. It will do the following steps:

        1. First retry after calling `torch.cuda.empty_cache()`.
        2. If that still fails, it will then retry by converting inputs
          to FP16.
        3. If that still fails trying to convert inputs to CPUs.
          In this case, it expects the function to dispatch to
          CPU implementation.

    Args:
        to_cpu (bool): Whether to convert outputs to CPU if get an OOM
            error. This will slow down the code significantly.
            Defaults to True.
        test (bool): Skip `_ignore_torch_cuda_oom` operate that can use
            lightweight data in unit test, only used in
            test unit. Defaults to False.

    Examples:
        >>> from mmdet.utils.memory import AvoidOOM
        >>> AvoidCUDAOOM = AvoidOOM()
        >>> output = AvoidOOM.retry_if_cuda_oom(
        >>>     some_torch_function)(input1, input2)
        >>> # To use as a decorator
        >>> # from mmdet.utils import AvoidCUDAOOM
        >>> @AvoidCUDAOOM.retry_if_cuda_oom
        >>> def function(*args, **kwargs):
        >>>     return None
    ```

    Note:
        1. The output may be on CPU even if inputs are on GPU. Processing
            on CPU will slow down the code significantly.
        2. When converting inputs to CPU, it will only look at each argument
            and check if it has `.device` and `.to` for conversion. Nested
            structures of tensors are not supported.
        3. Since the function might be called more than once, it has to be
            stateless.
    """

    def __init__(self, to_cpu=True, test=False):
        self.to_cpu = to_cpu
        self.test = test

    def retry_if_cuda_oom(self, func):
        """Makes a function retry itself after encountering pytorch's CUDA OOM
        error.

        The implementation logic is referred to
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/memory.py

        Args:
            func: a stateless callable that takes tensor-like objects
                as arguments.
        Returns:
            func: a callable which retries `func` if OOM is encountered.
        """  # noqa: W605

        @wraps(func)
        def wrapped(*args, **kwargs):

            # raw function
            if not self.test:
                with _ignore_torch_cuda_oom():
                    return func(*args, **kwargs)

                # Clear cache and retry
                torch.cuda.empty_cache()
                with _ignore_torch_cuda_oom():
                    return func(*args, **kwargs)

            # get the type and device of first tensor
            dtype, device = None, None
            values = args + tuple(kwargs.values())
            for value in values:
                if isinstance(value, torch.Tensor):
                    dtype = value.dtype
                    device = value.device
                    break
            if dtype is None or device is None:
                raise ValueError('There is no tensor in the inputs, '
                                 'cannot get dtype and device.')

            # Convert to FP16
            fp16_args = cast_tensor_type(args, dst_type=torch.half)
            fp16_kwargs = cast_tensor_type(kwargs, dst_type=torch.half)
            logger = MMLogger.get_current_instance()
            logger.warning(f'Attempting to copy inputs of {str(func)} '
                           'to FP16 due to CUDA OOM')

            # get input tensor type, the output type will same as
            # the first parameter type.
            with _ignore_torch_cuda_oom():
                output = func(*fp16_args, **fp16_kwargs)
                output = cast_tensor_type(
                    output, src_type=torch.half, dst_type=dtype)
                if not self.test:
                    return output
            logger.warning('Using FP16 still meet CUDA OOM')

            # Try on CPU. This will slow down the code significantly,
            # therefore print a notice.
            if self.to_cpu:
                logger.warning(f'Attempting to copy inputs of {str(func)} '
                               'to CPU due to CUDA OOM')
                cpu_device = torch.empty(0).device
                cpu_args = cast_tensor_type(args, dst_type=cpu_device)
                cpu_kwargs = cast_tensor_type(kwargs, dst_type=cpu_device)

                # convert outputs to GPU
                with _ignore_torch_cuda_oom():
                    logger.warning(f'Convert outputs to GPU (device={device})')
                    output = func(*cpu_args, **cpu_kwargs)
                    output = cast_tensor_type(
                        output, src_type=cpu_device, dst_type=device)
                    return output

                warnings.warn('Cannot convert output to GPU due to CUDA OOM, '
                              'the output is now on CPU, which might cause '
                              'errors if the output need to interact with GPU '
                              'data in subsequent operations')
                logger.warning('Cannot convert output to GPU due to '
                               'CUDA OOM, the output is on CPU now.')

                return func(*cpu_args, **cpu_kwargs)
            else:
                # may still get CUDA OOM error
                return func(*args, **kwargs)

        return wrapped

