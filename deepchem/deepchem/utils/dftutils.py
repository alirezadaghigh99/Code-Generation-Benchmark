class KSCalc(object):
    """
    Interface to DQC's KS calculation.

    Parameters
    ----------
    qc: BaseQCCalc
        object often acts as a wrapper around an engine class (from dqc.qccalc) that contains information about the self-consistent iterations.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation functional from nature with fully differentiable density functional theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/master/dqc/qccalc/ks.py
    """

    def __init__(self, qc: "BaseQCCalc"):
        self.qc = qc

    def energy(self) -> torch.Tensor:
        """
        Returns
        -------
        The total energy of the Kohn-Sham calculation for a particular system.
        """
        return self.qc.energy()

    def aodmtot(self) -> torch.Tensor:
        """
        Both interacting and non-interacting system's total energy can be expressed in terms of the density matrix. The ground state properties of a system can be calculated by minimizing the energy w.r.t the density matrix.

        Returns
        -------
        The total density matrix in atomic orbital bases.
        """
        dm = self.qc.aodm()
        if isinstance(dm, SpinParam):
            dmtot = dm.u + dm.d
        else:
            dmtot = dm
        return dmtot

    def dens(self, rgrid: torch.Tensor) -> torch.Tensor:
        """
        The ground state density n(r) of a system.

        Parameters
        ----------
        rgrid: torch.Tensor
            Calculate integration grid using dqc.grid.

        Returns
        -------
        The total density profile in the given grid

        Reference
        ---------
        https://github.com/diffqc/dqc/blob/master/dqc/grid/base_grid.py
        """
        dmtot = self.aodmtot()
        return self.qc.get_system().get_hamiltonian().aodm2dens(dmtot, rgrid)

    def force(self) -> torch.Tensor:
        """
        The force on an atom is calculated as the gradient of energy with respect to the atomic position.

        Returns
        -------
        The force for each atom.
        """
        ene = self.energy()
        atompos = self.qc.get_system().atompos
        is_grad_enabled = torch.is_grad_enabled()
        f, = torch.autograd.grad(ene,
                                 atompos,
                                 create_graph=is_grad_enabled,
                                 retain_graph=True)
        return f

class SpinParam(Generic[T]):
    """
    Data structure to store different values for spin-up and spin-down electrons.

    Attributes
    ----------
    u: any type
        The parameters that corresponds to the spin-up electrons.
    d: any type
        The parameters that corresponds to the spin-down electrons.

    References
    ----------
    Kasim, Muhammad F., and Sam M. Vinko. "Learning the exchange-correlation
    functional from nature with fully differentiable density functional
    theory." Physical Review Letters 127.12 (2021): 126403.
    https://github.com/diffqc/dqc/blob/master/dqc/utils/datastruct.py
    """

    u: T
    d: T

    def sum(self):
        """
        Returns the sum of up and down parameters
        """

        return self.u + self.d

    def reduce(self, fcn: Callable) -> T:
        """
        Reduce up and down parameters with the given function
        """

        return fcn(self.u, self.d)

