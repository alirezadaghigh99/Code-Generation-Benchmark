class VideoWriter(VideoReader):
    def __init__(self, video_path, codec="h264", dpi=100, fps=None):
        super(VideoWriter, self).__init__(video_path)
        self.codec = codec
        self.dpi = dpi
        if fps:
            self.fps = fps

    def set_bbox(self, x1, x2, y1, y2, relative=False):
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Coordinates look wrong... " f"Ensure {x1} < {x2} and {y1} < {y2}."
            )
        if not relative:
            x1 /= self._width
            x2 /= self._width
            y1 /= self._height
            y2 /= self._height
        bbox = x1, x2, y1, y2
        if any(coord > 1 for coord in bbox):
            warnings.warn(
                "Bounding box larger than the video... " "Clipping to video dimensions."
            )
            bbox = tuple(map(lambda x: min(x, 1), bbox))
        self._bbox = bbox

    def shorten(
        self, start, end, suffix="short", dest_folder=None, validate_inputs=True
    ):
        """
        Shorten the video from start to end.

        Parameter
        ----------
        start: str
            Time formatted in hours:minutes:seconds, where shortened video shall start.

        end: str
            Time formatted in hours:minutes:seconds, where shortened video shall end.

        suffix: str, optional
            String added to the name of the shortened video ('short' by default).

        dest_folder: str, optional
            Folder the video is saved into (by default, same as the original video)

        Returns
        -------
        str
            Full path to the shortened video
        """

        def validate_timestamp(stamp):
            if not isinstance(stamp, str):
                raise ValueError(
                    "Timestamp should be a string formatted "
                    "as hours:minutes:seconds."
                )
            time = datetime.datetime.strptime(stamp, "%H:%M:%S").time()
            # The above already raises a ValueError if formatting is wrong
            seconds = (time.hour * 60 + time.minute) * 60 + time.second
            if seconds > self.calc_duration():
                raise ValueError("Timestamps must not exceed the video duration.")

        if validate_inputs:
            for stamp in start, end:
                validate_timestamp(stamp)

        output_path = self.make_output_path(suffix, dest_folder)
        command = (
            f'ffmpeg -n -i "{self.video_path}" -ss {start} -to {end} '
            f'-c:a copy "{output_path}"'
        )
        subprocess.call(command, shell=True)
        return output_path

    def split(self, n_splits, suffix="split", dest_folder=None):
        """
        Split a video into several shorter ones of equal duration.

        Parameters
        ----------
        n_splits : int
            Number of shorter videos to produce

        suffix: str, optional
            String added to the name of the splits ('short' by default).

        dest_folder: str, optional
            Folder the video splits are saved into (by default, same as the original video)

        Returns
        -------
        list
            Paths of the video splits
        """
        if not n_splits > 1:
            raise ValueError("The video should at least be split in half.")
        chunk_dur = self.calc_duration() / n_splits
        splits = np.arange(n_splits + 1) * chunk_dur
        time_formatter = lambda val: str(datetime.timedelta(seconds=val))
        clips = []
        for n, (start, end) in enumerate(zip(splits, splits[1:]), start=1):
            clips.append(
                self.shorten(
                    time_formatter(start),
                    time_formatter(end),
                    f"{suffix}{n}",
                    dest_folder,
                    validate_inputs=False,
                )
            )
        return clips

    def crop(self, suffix="crop", dest_folder=None):
        x1, _, y1, _ = self.get_bbox()
        output_path = self.make_output_path(suffix, dest_folder)
        command = (
            f'ffmpeg -n -i "{self.video_path}" '
            f"-filter:v crop={self.width}:{self.height}:{x1}:{y1} "
            f'-c:a copy "{output_path}"'
        )
        subprocess.call(command, shell=True)
        return output_path

    def rescale(
        self,
        width,
        height=-1,
        rotatecw="No",
        angle=0.0,
        suffix="rescale",
        dest_folder=None,
    ):
        output_path = self.make_output_path(suffix, dest_folder)
        command = (
            f'ffmpeg -n -i "{self.video_path}" -filter:v '
            f'"scale={width}:{height}{{}}" -c:a copy "{output_path}"'
        )
        # Rotate, see: https://stackoverflow.com/questions/3937387/rotating-videos-with-ffmpeg
        # interesting option to just update metadata.
        if rotatecw == "Arbitrary":
            angle = np.deg2rad(angle)
            command = command.format(f", rotate={angle}")
        elif rotatecw == "Yes":
            command = command.format(f", transpose=1")
        else:
            command = command.format("")
        subprocess.call(command, shell=True)
        return output_path

    @staticmethod
    def write_frame(frame, where):
        cv2.imwrite(where, frame[..., ::-1])

    def make_output_path(self, suffix, dest_folder):
        if not dest_folder:
            dest_folder = self.directory
        return os.path.join(dest_folder, f"{self.name}{suffix}{self.format}")