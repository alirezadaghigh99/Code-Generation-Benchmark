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