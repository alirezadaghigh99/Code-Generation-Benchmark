    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        The number of samples is determined by the value of ``Device.shots``,
        which can be directly modified.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Raises:
            NotImplementedError: if the device does not support sampling

        Returns:
            array[float]: samples in an array of dimension ``(shots,)``
        """
        raise NotImplementedError(
            f"Returning samples from QNodes not currently supported by {self.short_name}"
        )    def sample(self, observable, wires, par):
        """Return a sample of an observable.

        The number of samples is determined by the value of ``Device.shots``,
        which can be directly modified.

        Note: all arguments support _lists_, which indicate a tensor
        product of observables.

        Args:
            observable (str or list[str]): name of the observable(s)
            wires (Wires): wires the observable(s) is to be measured on
            par (tuple or list[tuple]]): parameters for the observable(s)

        Raises:
            NotImplementedError: if the device does not support sampling

        Returns:
            array[float]: samples in an array of dimension ``(shots,)``
        """
        raise NotImplementedError(
            f"Returning samples from QNodes not currently supported by {self.short_name}"
        )