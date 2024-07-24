class device:  # noqa: N801 invalid-class-name # pyright: ignore [reportIncompatibleVariableOverride]
        def __init__(self, device) -> None:
            self.device = device

class Event(
        device_interface._EventBase
    ):  # pyright: ignore [reportPrivateImportUsage]
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
        ) -> None:
            self.enable_timing = enable_timing
            self.recorded_time: int | None = None

        def record(self, stream) -> None:
            if not self.enable_timing:
                return
            assert self.recorded_time is None
            self.recorded_time = time.perf_counter_ns()

        def elapsed_time(self, end_event: DeviceInterface.Event) -> float:
            assert self.recorded_time
            assert end_event.recorded_time
            # convert to ms
            return (end_event.recorded_time - self.recorded_time) / 1000000

        def wait(self, stream) -> None:
            pass

        def query(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

class Event(
        device_interface._EventBase
    ):  # pyright: ignore [reportPrivateImportUsage]
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
        ) -> None:
            self.enable_timing = enable_timing
            self.recorded_time: int | None = None

        def record(self, stream) -> None:
            if not self.enable_timing:
                return
            assert self.recorded_time is None
            self.recorded_time = time.perf_counter_ns()

        def elapsed_time(self, end_event: DeviceInterface.Event) -> float:
            assert self.recorded_time
            assert end_event.recorded_time
            # convert to ms
            return (end_event.recorded_time - self.recorded_time) / 1000000

        def wait(self, stream) -> None:
            pass

        def query(self) -> None:
            pass

        def synchronize(self) -> None:
            pass

