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

class EGLContext:
    """
    A class representing an EGL context. In short, EGL allows us to render OpenGL con-
    tent in a headless mode, that is without an actual display to render to. This capa-
    bility enables MeshRasterizerOpenGL to render on the GPU and then transfer the re-
    sults to PyTorch3D.
    """

    def __init__(self, width: int, height: int, cuda_device_id: int = 0) -> None:
        """
        Args:
            width: Width of the "display" to render to.
            height: Height of the "display" to render to.
            cuda_device_id: Device ID to render to, in the CUDA convention (note that
                this might be different than EGL's device numbering).
        """
        # Lock used to prevent multiple threads from rendering on the same device
        # at the same time, creating/destroying contexts at the same time, etc.
        self.lock = threading.Lock()
        self.cuda_device_id = cuda_device_id
        self.device = _get_cuda_device(self.cuda_device_id)
        self.width = width
        self.height = height
        self.dpy = egl.eglGetPlatformDisplayEXT(
            EGL_PLATFORM_DEVICE_EXT, self.device, None
        )
        major, minor = egl.EGLint(), egl.EGLint()

        # Initialize EGL components: the display, surface, and context
        egl.eglInitialize(self.dpy, ctypes.pointer(major), ctypes.pointer(minor))

        config = _get_egl_config(self.dpy, egl.EGL_PBUFFER_BIT)
        pb_surf_attribs = _egl_convert_to_int_array(
            {
                egl.EGL_WIDTH: width,
                egl.EGL_HEIGHT: height,
            }
        )
        self.surface = egl.eglCreatePbufferSurface(self.dpy, config, pb_surf_attribs)
        if self.surface == egl.EGL_NO_SURFACE:
            raise RuntimeError("Failed to create an EGL surface.")

        if not egl.eglBindAPI(egl.EGL_OPENGL_API):
            raise RuntimeError("Failed to bind EGL to the OpenGL API.")
        self.context = egl.eglCreateContext(self.dpy, config, egl.EGL_NO_CONTEXT, None)
        if self.context == egl.EGL_NO_CONTEXT:
            raise RuntimeError("Failed to create an EGL context.")

    @contextlib.contextmanager
    def active_and_locked(self):
        """
        A context manager used to make sure a given EGL context is only current in
        a single thread at a single time. It is recommended to ALWAYS use EGL within
        a `with context.active_and_locked():` context.

        Throws:
            EGLError when the context cannot be made current or make non-current.
        """
        self.lock.acquire()
        egl.eglMakeCurrent(self.dpy, self.surface, self.surface, self.context)
        try:
            yield
        finally:
            egl.eglMakeCurrent(
                self.dpy, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT
            )
            self.lock.release()

    def get_context_info(self) -> Dict[str, Any]:
        """
        Return context info. Useful for debugging.

        Returns:
            A dict of keys and ints, representing the context's display, surface,
            the context itself, and the current thread.
        """
        return {
            "dpy": self.dpy,
            "surface": self.surface,
            "context": self.context,
            "thread": threading.get_ident(),
        }

    def release(self):
        """
        Release the context's resources.
        """
        self.lock.acquire()
        try:
            if self.surface:
                egl.eglDestroySurface(self.dpy, self.surface)
            if self.context and self.dpy:
                egl.eglDestroyContext(self.dpy, self.context)
            egl.eglMakeCurrent(
                self.dpy, egl.EGL_NO_SURFACE, egl.EGL_NO_SURFACE, egl.EGL_NO_CONTEXT
            )
            if self.dpy:
                egl.eglTerminate(self.dpy)
        except EGLError as err:
            print(
                f"EGL could not release context on device cuda:{self.cuda_device_id}."
                " This can happen if you created two contexts on the same device."
                " Instead, you can use DeviceContextStore to use a single context"
                " per device, and EGLContext.make_(in)active_in_current_thread to"
                " (in)activate the context as needed."
            )
            raise err

        egl.eglReleaseThread()
        self.lock.release()

