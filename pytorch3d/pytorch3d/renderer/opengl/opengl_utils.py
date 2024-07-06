def get_egl_context(self, device):
        """
        Return an EGL context on a given CUDA device. If we have not created such a
        context yet, create a new one and store it in a dict. The context if not current
        (you should use the `with egl_context.active_and_locked:` context manager when
        you need it to be current). This function is thread-safe.

        Args:
            device: A torch.device.

        Returns: An EGLContext on the requested device. The context will have size
            self.max_egl_width and self.max_egl_height.
        """
        cuda_device_id = device.index
        with self._lock:
            egl_context = self._egl_contexts.get(cuda_device_id, None)
            if egl_context is None:
                self._egl_contexts[cuda_device_id] = EGLContext(
                    self.max_egl_width, self.max_egl_height, cuda_device_id
                )
            return self._egl_contexts[cuda_device_id]

