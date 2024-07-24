class GenericSolver(BaseSolver):
    def get_solution(self, copy=True, best=True):
        r"""Get a (callable) solution object. See this usage example:

        .. code-block:: python3

            solution = solver.get_solution()
            point_coords = train_generator.get_examples()
            value_at_points = solution(point_coords)

        :param copy:
            Whether to make a copy of the networks so that subsequent training doesn't affect the solution;
            Defaults to True.
        :type copy: bool
        :param best:
            Whether to return the solution with lowest loss instead of the solution after the last epoch.
            Defaults to True.
        :type best: bool
        :return:
            A solution object which can be called.
            To evaluate the solution on certain points,
            you should pass the coordinates vector(s) to the returned solution.
        :rtype: BaseSolution
        """
        nets = self.best_nets if best else self.nets
        conditions = self.conditions
        if copy:
            nets = deepcopy(nets)
            conditions = deepcopy(conditions)

        return GenericSolution(nets, conditions)

