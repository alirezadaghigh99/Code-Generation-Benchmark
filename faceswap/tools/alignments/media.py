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