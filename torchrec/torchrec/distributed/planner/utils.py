class LuusJaakolaSearch:
    """Implements a clamped variant of Luus Jaakola search.

    See https://en.wikipedia.org/wiki/Luus-Jaakola.
    """

    def __init__(
        self,
        A: float,
        B: float,
        max_iterations: int,
        seed: int = 42,
        left_cost: Optional[float] = None,
    ) -> None:
        self.left = A
        self.right = B
        self.iteration = -1
        self.max_iterations = max_iterations

        self.gen = torch.Generator()
        self.gen.manual_seed(seed)

        self.x: float = self.uniform(self.left, self.right)
        self.fx: float = 0.0
        self.y: float = math.nan
        self.fleft: Optional[float] = left_cost
        self.fright: Optional[float] = None
        self.d: float = self.right - self.left

    def shrink_right(self, B: float) -> None:
        "Shrink right boundary given [B,infinity) -> infinity"
        self.right = B
        self.fright = math.inf
        self.d = self.right - self.left
        self.x = self.clamp(self.x)

    def clamp(self, x: float) -> float:
        "Clamp x into range [left, right]"
        if x < self.left:
            return self.left
        if x > self.right:
            return self.right
        return x

    def uniform(self, A: float, B: float) -> float:
        "Return a random uniform position in range [A,B]."
        u = torch.rand(1, generator=self.gen).item()
        return A + (B - A) * u

    def next(self, fy: float) -> Optional[float]:
        """Return the next probe point 'y' to evaluate, given the previous result.

        The first time around fy is ignored. Subsequent invocations should provide the
        result of evaluating the function being minimized, i.e. f(y).

        Returns None when the maximum number of iterations has been reached.
        """
        self.iteration += 1
        if self.iteration == 0:
            return self.x
        elif self.iteration == 1:
            self.fx = fy
        elif self.iteration == self.max_iterations:
            return None
        elif fy <= self.fx:
            self.x = self.y
            self.fx = fy
            self.d = 0.95 * self.d

        if self.y == self.left:
            self.fleft = fy
        elif self.y == self.right:
            self.fright = fy

        while True:
            a = self.uniform(-self.d, self.d)
            y = self.clamp(self.x + a)
            # Unlike standard Luus-Jaakola, we don't want to explore outside of our bounds.
            # Clamping can cause us to explore the boundary multiple times, so we
            # remember if we already know the boundary cost and request a new sample if
            # we do.
            if y == self.left and self.fleft is not None:
                continue
            if y == self.right and self.fright is not None:
                continue
            self.y = y
            return self.y

    def best(self) -> Tuple[float, float]:
        "Return the best position so far, and its associated cost."
        return self.x, self.fx

