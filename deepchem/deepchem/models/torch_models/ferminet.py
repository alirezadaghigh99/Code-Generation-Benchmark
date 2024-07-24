class FerminetModel(TorchModel):
    """A deep-learning based Variational Monte Carlo method [1]_ for calculating the ab-initio
    solution of a many-electron system.

    This model aims to calculate the ground state energy of a multi-electron system
    using a baseline solution as the Hartree-Fock. An MCMC technique is used to sample
    electrons and DNNs are used to caluclate the square magnitude of the wavefunction,
    in which electron-electron repulsions also are included in the calculation(in the
    form of Jastrow factor envelopes). The model requires only the nucleus' coordinates
    as input.

    This method is based on the following paper:

    Example
    -------
    >>> from deepchem.models.torch_models.ferminet import FerminetModel
    >>> H2_molecule = [['H', [0, 0, 0]], ['H', [0, 0, 0.748]]]
    >>> mol = FerminetModel(H2_molecule, spin=0, ion_charge=0, tasks='pretraining') # doctest: +IGNORE_RESULT
    converged SCF energy = -0.895803169899509  <S^2> = 0  2S+1 = 1
     ** Mulliken pop alpha/beta on meta-lowdin orthogonal AOs **
     ** Mulliken pop       alpha | beta **
    pop of  0 H 1s        0.50000 | 0.50000
    pop of  1 H 1s        0.50000 | 0.50000
    In total             1.00000 | 1.00000
     ** Mulliken atomic charges   ( Nelec_alpha | Nelec_beta ) **
    charge of  0H =      0.00000  (     0.50000      0.50000 )
    charge of  1H =      0.00000  (     0.50000      0.50000 )
    converged SCF energy = -0.895803169899509  <S^2> = 0  2S+1 = 1
    >>> mol.train(nb_epoch=3)
    >>> print(mol.model.psi_up.size())
    torch.Size([8, 16, 1, 1])

    References
    ----------
    .. [1] Spencer, James S., et al. Better, Faster Fermionic Neural Networks. arXiv:2011.07125, arXiv, 13 Nov. 2020. arXiv.org, http://arxiv.org/abs/2011.07125.

    Note
    ----
    This class requires pySCF to be installed.
    """.replace('+IGNORE_RESULT', '+ELLIPSIS\n<...>')

    def __init__(self,
                 nucleon_coordinates: List[List],
                 spin: int,
                 ion_charge: int,
                 seed: Optional[int] = None,
                 batch_no: int = 8,
                 random_walk_steps: int = 10,
                 steps_per_update: int = 10,
                 tasks: str = 'pretraining'):
        """
    Parameters:
    -----------
    nucleon_coordinates: List[List]
      A list containing nucleon coordinates as the values with the keys as the element's symbol.
    spin: int
      The total spin of the molecule system.
    ion_charge: int
      The total charge of the molecule system.
    seed_no: int, optional (default None)
      Random seed to use for electron initialization.
    batch_no: int, optional (default 8)
      Number of batches of the electron's positions to be initialized.
    random_walk_steps: int (default 10)
        Number of random walk steps to be performed in a single move.
    steps_per_update: int (default: 10)
        Number of steps after which the electron sampler should update the electron parameters.
    tasks: str (default: 'pretraining')
        The type of task to be performed - 'pretraining', 'training'

    Attributes:
    -----------
    nucleon_pos: np.ndarray
        numpy array value of nucleon_coordinates
    electron_no: np.ndarray
        Torch tensor containing electrons for each atom in the nucleus
    molecule: ElectronSampler
        ElectronSampler object which performs MCMC and samples electrons
    loss_value: torch.Tensor (default torch.tensor(0))
        torch tensor storing the loss value from the last iteration
    """
        self.nucleon_coordinates = nucleon_coordinates
        self.seed = seed
        self.batch_no = batch_no
        self.spin = spin
        self.ion_charge = ion_charge
        self.batch_no = batch_no
        self.random_walk_steps = random_walk_steps
        self.steps_per_update = steps_per_update
        self.loss_value: torch.Tensor = torch.tensor(0)
        self.tasks = tasks

        no_electrons = []
        nucleons = []

        table = Chem.GetPeriodicTable()
        index = 0
        for i in self.nucleon_coordinates:
            atomic_num = table.GetAtomicNumber(i[0])
            no_electrons.append([atomic_num])
            nucleons.append(i[1])
            index += 1

        self.electron_no: np.ndarray = np.array(no_electrons)
        charge: np.ndarray = self.electron_no.reshape(
            np.shape(self.electron_no)[0])
        self.nucleon_pos: np.ndarray = np.array(nucleons)

        # Initialization for ionic molecules
        if np.sum(self.electron_no) < self.ion_charge:
            raise ValueError("Given charge is not initializable")

        total_electrons = np.sum(self.electron_no) - self.ion_charge

        if self.spin >= 0:
            self.up_spin = (total_electrons + 2 * self.spin) // 2
            self.down_spin = total_electrons - self.up_spin
        else:
            self.down_spin = (total_electrons - 2 * self.spin) // 2
            self.up_spin = total_electrons - self.down_spin

        if self.up_spin - self.down_spin != self.spin:
            raise ValueError("Given spin is not feasible")

        nucl = torch.from_numpy(self.nucleon_pos)
        self.model = Ferminet(nucl,
                              spin=(self.up_spin, self.down_spin),
                              nuclear_charge=torch.Tensor(charge),
                              batch_size=self.batch_no)

        self.molecule: ElectronSampler = ElectronSampler(
            batch_no=self.batch_no,
            central_value=self.nucleon_pos,
            seed=self.seed,
            f=lambda x: self.random_walk(x
                                        ),  # Will be replaced in successive PR
            steps=self.random_walk_steps,
            steps_per_update=self.steps_per_update
        )  # sample the electrons using the electron sampler
        self.molecule.gauss_initialize_position(
            self.electron_no,
            stddev=1.0)  # initialize the position of the electrons
        self.prepare_hf_solution()
        super(FerminetModel, self).__init__(
            self.model,
            loss=torch.nn.MSELoss())  # will update the loss in successive PR

    def evaluate_hf(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to calculate orbital values at sampled electron's position.

        Parameters:
        -----------
        x: np.ndarray
            Contains the sampled electrons coordinates in a numpy array.

        Returns:
        --------
        2 numpy arrays containing the up-spin and down-spin orbitals in a numpy array respectively.
        """
        x = np.reshape(x, [-1, 3 * (self.up_spin + self.down_spin)])
        leading_dims = x.shape[:-1]
        x = np.reshape(x, [-1, 3])
        coeffs = self.mf.mo_coeff
        gto_op = 'GTOval_sph'
        ao_values = self.mol.eval_gto(gto_op, x)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
        mo_values_list = [
            np.reshape(mo, leading_dims + (self.up_spin + self.down_spin, -1))
            for mo in mo_values
        ]
        return mo_values_list[0][
            ..., :self.up_spin, :self.up_spin], mo_values_list[1][
                ..., self.up_spin:, :self.down_spin]

    def prepare_hf_solution(self):
        """Prepares the HF solution for the molecule system which is to be used in pretraining
        """
        try:
            import pyscf
        except ModuleNotFoundError:
            raise ImportError("This module requires pySCF")

        molecule = ""
        for i in range(len(self.nucleon_pos)):
            molecule = molecule + self.nucleon_coordinates[i][0] + " " + str(
                self.nucleon_coordinates[i][1][0]) + " " + str(
                    self.nucleon_coordinates[i][1][1]) + " " + str(
                        self.nucleon_coordinates[i][1][2]) + ";"
        self.mol = pyscf.gto.Mole(atom=molecule, basis='sto-6g')
        self.mol.parse_arg = False
        self.mol.unit = 'Bohr'
        self.mol.spin = (self.up_spin - self.down_spin)
        self.mol.charge = self.ion_charge
        self.mol.build(parse_arg=False)
        self.mf = pyscf.scf.UHF(self.mol)
        self.mf.run()
        dm = self.mf.make_rdm1()
        _, chg = pyscf.scf.uhf.mulliken_meta(self.mol, dm)
        excess_charge = np.array(chg)
        tmp_charge = self.ion_charge
        while tmp_charge != 0:
            if (tmp_charge < 0):
                charge_index = np.argmin(excess_charge)
                tmp_charge += 1
                self.electron_no[charge_index] += 1
                excess_charge[charge_index] += 1
            elif (tmp_charge > 0):
                charge_index = np.argmax(excess_charge)
                tmp_charge -= 1
                self.electron_no[charge_index] -= 1
                excess_charge[charge_index] -= 1

        self.molecule.gauss_initialize_position(
            self.electron_no,
            stddev=2.0)  # initialize the position of the electrons
        _ = self.mf.kernel()

    def random_walk(self, x: np.ndarray):
        """
        Function to be passed on to electron sampler for random walk and gets called at each step of sampling

        Parameters
        ----------
        x: np.ndarray
            contains the sampled electrons coordinate in the shape (batch_size,number_of_electrons*3)

        Returns:
        --------
        A numpy array containing the joint probability of the hartree fock and the sampled electron's position coordinates
        """
        x_torch = torch.from_numpy(x).view(self.batch_no, -1, 3)
        if self.tasks == 'pretraining':
            x_torch.requires_grad = True
        else:
            x_torch.requires_grad = False
        output = self.model.forward(x_torch)
        np_output = output.detach().cpu().numpy()

        if self.tasks == 'pretraining':
            up_spin_mo, down_spin_mo = self.evaluate_hf(x)
            hf_product = np.prod(
                np.diagonal(up_spin_mo, axis1=1, axis2=2), axis=1) * np.prod(
                    np.diagonal(down_spin_mo, axis1=1, axis2=2), axis=1)
            self.model.loss(up_spin_mo, down_spin_mo)
            np_output[:int(self.batch_no / 2)] = hf_product[:int(self.batch_no /
                                                                 2)]
            return 2 * np.log(np.abs(np_output))

        if self.tasks == 'burn':
            return 2 * np.log(np.abs(np_output))

        if self.tasks == 'training':
            energy = self.model.loss(pretrain=[False])
            self.energy_sampled: torch.Tensor = torch.cat(
                (self.energy_sampled, energy.unsqueeze(0)))
            return 2 * np.log(np.abs(np_output))

    def prepare_train(self, burn_in: int = 100):
        """
        Function to perform burn-in and to change the model parameters for training.

        Parameters
        ----------
        burn_in:int (default: 100)
            number of steps for to perform burn-in before the aactual training.
        """
        self.tasks = 'burn'
        self.molecule.gauss_initialize_position(self.electron_no, stddev=1.0)
        tmp_x = self.molecule.x
        for _ in range(burn_in):
            self.molecule.move(stddev=0.02)
            self.molecule.x = tmp_x
        self.molecule.move(stddev=0.02)
        self.tasks = 'training'

    def train(self,
              nb_epoch: int = 200,
              lr: float = 0.002,
              weight_decay: float = 0,
              std: float = 0.08,
              std_init: float = 0.02,
              steps_std: int = 100):
        """
        function to run training or pretraining.

        Parameters
        ----------
        nb_epoch: int (default: 200)
            contains the number of pretraining steps to be performed
        lr : float (default: 0.002)
            contains the learning rate for the model fitting
        weight_decay: float (default: 0.002)
            contains the weight_decay for the model fitting
        std: float (default: 0.08)
            The standard deviation for the electron update during training
        std_init: float (default: 0.02)
            The standard deviation for the electron update during pretraining
        steps_std: float (default 100)
            The number of steps for standard deviation increase
        """

        # hook function below is an efficient way modifying the gradients on the go rather than looping
        def energy_hook(grad, random_walk_steps):
            """
            hook function to modify the gradients
            """
            # using non-local variables as a means of parameter passing
            nonlocal energy_local, energy_mean
            new_grad = (2 / random_walk_steps) * (
                (energy_local - energy_mean) * grad)
            return new_grad.float()

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        if (self.tasks == 'pretraining'):
            for iteration in range(nb_epoch):
                optimizer.zero_grad()
                accept = self.molecule.move(stddev=std_init)
                if iteration % steps_std == 0:
                    if accept > 0.55:
                        std_init *= 1.1
                    else:
                        std_init /= 1.1
                self.loss_value = (torch.mean(self.model.running_diff) /
                                   self.random_walk_steps)
                self.loss_value.backward()
                optimizer.step()
                logging.info("The loss for the pretraining iteration " +
                             str(iteration) + " is " +
                             str(self.loss_value.item()))
                self.model.running_diff = torch.zeros(self.batch_no)

        if (self.tasks == 'training'):
            energy_local = None
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
            self.final_energy = torch.tensor(0.0)
            with torch.no_grad():
                hooks = list(
                    map(
                        lambda param: param.
                        register_hook(lambda grad: energy_hook(
                            grad, self.random_walk_steps)),
                        self.model.parameters()))
            for iteration in range(nb_epoch):
                optimizer.zero_grad()
                self.energy_sampled = torch.tensor([])
                # the move function calculates the energy of sampled electrons and samples new set of electrons (does not calculate loss)
                accept = self.molecule.move(stddev=std)
                if iteration % steps_std == 0:
                    if accept > 0.55:
                        std_init *= 1.2
                    else:
                        std_init /= 1.2
                median, _ = torch.median(self.energy_sampled, 0)
                variance = torch.mean(torch.abs(self.energy_sampled - median))
                # clipping local energies which are away 5 times the variance from the median
                clamped_energy = torch.clamp(self.energy_sampled,
                                             max=median + 5 * variance,
                                             min=median - 5 * variance)
                energy_mean = torch.mean(clamped_energy)
                logging.info("The mean energy for the training iteration " +
                             str(iteration) + " is " + str(energy_mean.item()))
                self.final_energy = self.final_energy + energy_mean
                # using the sampled electrons from the electron sampler for bacward pass and modifying gradients
                sample_history = torch.from_numpy(
                    self.molecule.sampled_electrons).view(
                        self.random_walk_steps, self.batch_no, -1, 3)
                optimizer.zero_grad()
                for i in range(self.random_walk_steps):
                    # going through each step of random walk and calculating the modified gradients with local energies
                    input_electron = sample_history[i]
                    input_electron.requires_grad = True
                    energy_local = torch.mean(clamped_energy[i])
                    self.model.forward(input_electron)
                    self.loss_value = torch.mean(
                        torch.log(torch.abs(self.model.psi)))
                    self.loss_value.backward()
                optimizer.step()
            self.final_energy = self.final_energy / nb_epoch
            list(map(lambda hook: hook.remove(), hooks))

