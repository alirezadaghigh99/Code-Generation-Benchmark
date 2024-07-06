def make_device(self, set_cuda_device: bool = False) -> torch.device:
        """
        Returns (an optionally sets) the default `torch.device` to use.

        Automatically called from :meth:`init_distributed`.

        :param set_cuda_device: If True, sets the default device
            by calling :func:`torch.cuda.set_device`.
        :return: The default device to be used for `torch.distributed`
            communications.
        """
        if self.is_distributed:
            device_id = self.rank
        else:
            device_id = 0

        if self.use_cuda and device_id >= 0:
            ref_device = torch.device(f"cuda:{device_id}")
            if set_cuda_device:
                torch.cuda.set_device(ref_device)
        else:
            ref_device = torch.device("cpu")
        return ref_device

