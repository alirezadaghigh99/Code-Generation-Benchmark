    def get_group(self, ranks_in_group: List[int], backend: Optional[str] = None) -> ProcessGroup:
        """Get the process group with the given ranks. It the process group doesn't exist, it will be created.

        Args:
            ranks_in_group (List[int]): Ranks in the process group.
            backend (Optional[str], optional): Backend of the process group. Defaults to None.

        Returns:
            ProcessGroup: The process group with the given ranks.
        """
        ranks_in_group = sorted(ranks_in_group)
        if tuple(ranks_in_group) not in self._group_to_ranks:
            group = dist.new_group(ranks_in_group, backend=backend)
            self._ranks_to_group[tuple(ranks_in_group)] = group
            self._group_to_ranks[group] = tuple(ranks_in_group)
        return self._ranks_to_group[tuple(ranks_in_group)]