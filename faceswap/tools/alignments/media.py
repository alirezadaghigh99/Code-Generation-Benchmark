class Frames(MediaLoader):
    """ Object to hold the frames that are to be checked against """

    def process_folder(self) -> Generator[dict[str, str], None, None]:
        """ Iterate through the frames folder pulling the base filename

        Yields
        ------
        dict
            The full framename, the filename and the file extension of the frame
        """
        iterator = self.process_video if self.is_video else self.process_frames
        for item in iterator():
            yield item

    def process_frames(self) -> Generator[dict[str, str], None, None]:
        """ Process exported Frames

        Yields
        ------
        dict
            The full framename, the filename and the file extension of the frame
        """
        logger.info("Loading file list from %s", self.folder)
        for frame in os.listdir(self.folder):
            if not self.valid_extension(frame):
                continue
            filename = os.path.splitext(frame)[0]
            file_extension = os.path.splitext(frame)[1]

            retval = {"frame_fullname": frame,
                      "frame_name": filename,
                      "frame_extension": file_extension}
            logger.trace(retval)  # type: ignore
            yield retval

    def process_video(self) -> Generator[dict[str, str], None, None]:
        """Dummy in frames for video

        Yields
        ------
        dict
            The full framename, the filename and the file extension of the frame
        """
        logger.info("Loading video frames from %s", self.folder)
        vidname, ext = os.path.splitext(os.path.basename(self.folder))
        for i in range(self.count):
            idx = i + 1
            # Keep filename format for outputted face
            filename = f"{vidname}_{idx:06d}"
            retval = {"frame_fullname": f"{filename}{ext}",
                      "frame_name": filename,
                      "frame_extension": ext}
            logger.trace(retval)  # type: ignore
            yield retval

    def load_items(self) -> dict[str, tuple[str, str]]:
        """ Load the frame info into dictionary

        Returns
        -------
        dict
            Fullname as key, tuple of frame name and extension as value
        """
        frames: dict[str, tuple[str, str]] = {}
        for frame in T.cast(list[dict[str, str]], self.file_list_sorted):
            frames[frame["frame_fullname"]] = (frame["frame_name"],
                                               frame["frame_extension"])
        logger.trace(frames)  # type: ignore
        return frames

    def sorted_items(self) -> list[dict[str, str]]:
        """ Return the items sorted by filename

        Returns
        -------
        list
            The sorted list of frame information
        """
        items = sorted(self.process_folder(), key=lambda x: (x["frame_name"]))
        logger.trace(items)  # type: ignore
        return items