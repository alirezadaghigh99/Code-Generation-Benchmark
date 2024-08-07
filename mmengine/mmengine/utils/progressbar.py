class ProgressBar:
    """A progress bar which can print the progress.

    Args:
        task_num (int): Number of total steps. Defaults to 0.
        bar_width (int): Width of the progress bar. Defaults to 50.
        start (bool): Whether to start the progress bar in the constructor.
            Defaults to True.
        file (callable): Progress bar output mode. Defaults to "sys.stdout".

    Examples:
        >>> import mmengine
        >>> import time
        >>> bar = mmengine.ProgressBar(10)
        >>> for i in range(10):
        >>>    bar.update()
        >>>    time.sleep(1)
    """

    def __init__(self,
                 task_num: int = 0,
                 bar_width: int = 50,
                 start: bool = True,
                 file=sys.stdout):
        self.task_num = task_num
        self.bar_width = bar_width
        self.completed = 0
        self.file = file
        if start:
            self.start()

    @property
    def terminal_width(self):
        width, _ = get_terminal_size()
        return width

    def start(self):
        if self.task_num > 0:
            self.file.write(f'[{" " * self.bar_width}] 0/{self.task_num}, '
                            'elapsed: 0s, ETA:')
        else:
            self.file.write('completed: 0, elapsed: 0s')
        self.file.flush()
        self.timer = Timer()

    def update(self, num_tasks: int = 1):
        """update progressbar.

        Args:
            num_tasks (int): Update step size.
        """
        assert num_tasks > 0
        self.completed += num_tasks
        elapsed = self.timer.since_start()
        if elapsed > 0:
            fps = self.completed / elapsed
        else:
            fps = float('inf')
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            msg = f'\r[{{}}] {self.completed}/{self.task_num}, ' \
                  f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, ' \
                  f'ETA: {eta:5}s'

            bar_width = min(self.bar_width,
                            int(self.terminal_width - len(msg)) + 2,
                            int(self.terminal_width * 0.6))
            bar_width = max(2, bar_width)
            mark_width = int(bar_width * percentage)
            bar_chars = '>' * mark_width + ' ' * (bar_width - mark_width)
            self.file.write(msg.format(bar_chars))
        else:
            self.file.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        self.file.flush()

