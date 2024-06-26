class ImagesCanvas(ttk.Frame):  # pylint:disable=too-many-ancestors
    """ tkinter Canvas that holds the preview images.

    Parameters
    ----------
    app: :class:`Preview`
        The main tkinter Preview app
    parent: tkinter object
        The parent tkinter object that holds the canvas
    """
    def __init__(self, app: Preview, parent: ttk.PanedWindow) -> None:
        logger.debug("Initializing %s: (app: %s, parent: %s)",
                     self.__class__.__name__, app, parent)
        super().__init__(parent)
        self.pack(expand=True, fill=tk.BOTH, padx=2, pady=2)

        self._display: FacesDisplay = parent.preview_display  # type: ignore
        self._canvas = tk.Canvas(self, bd=0, highlightthickness=0)
        self._canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._displaycanvas = self._canvas.create_image(0, 0,
                                                        image=self._display.tk_image,
                                                        anchor=tk.NW)
        self.bind("<Configure>", self._resize)
        logger.debug("Initialized %s", self.__class__.__name__)

    def _resize(self, event: tk.Event) -> None:
        """ Resize the image to fit the frame, maintaining aspect ratio """
        logger.debug("Resizing preview image")
        framesize = (event.width, event.height)
        self._display.set_display_dimensions(framesize)
        self.reload()

    def reload(self) -> None:
        """ Update the images in the canvas and redraw """
        logger.debug("Reloading preview image")
        self._display.update_tk_image()
        self._canvas.itemconfig(self._displaycanvas, image=self._display.tk_image)
        logger.debug("Reloaded preview image")