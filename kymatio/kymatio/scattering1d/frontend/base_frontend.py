class TimeFrequencyScatteringBase(ScatteringBase1D):
    def __init__(self, J, J_fr, Q, shape, T=None, stride=None,
            Q_fr=1, F=None, stride_fr=None,
            out_type='array', format='time', backend=None):
        max_order = 2
        oversampling = None
        super(TimeFrequencyScatteringBase, self).__init__(J, shape, Q, T,
            stride, max_order, oversampling, out_type, backend)
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self._F = F
        self.oversampling_fr = 0
        self._stride_fr = stride_fr
        self.format = format

    def build(self):
        super(TimeFrequencyScatteringBase, self).build()
        super(TimeFrequencyScatteringBase, self).create_filters()

        # check the number of filters per octave
        if np.any(np.array(self.Q_fr) < 1):
            raise ValueError('Q_fr must be >= 1, got {}'.format(self.Q_fr))

        if isinstance(self.Q_fr, int):
            self.Q_fr = (self.Q_fr,)
        elif isinstance(self.Q_fr, tuple):
            if (len(self.Q_fr) != 1):
                raise NotImplementedError("Q_fr must be an integer or 1-tuple. "
                                          "Time-frequency scattering "
                                          "beyond order 2 is not implemented.")
        else:
            raise ValueError("Q_fr must be an integer or 1-tuple.")

        # Compute the minimum support to pad (ideally)
        min_to_pad_fr = 8 * min(self.F, 2 ** self.J_fr)

        # We want to pad the frequency domain to the minimum number that is:
        # (1) greater than number of first-order coefficients, N_input_fr,
        #     by a margin of at least min_to_pad_fr
        # (2) a multiple of all subsampling factors of frequential scattering:
        #     2**1, 2**2, etc. up to 2**K_fr = (2**J_fr / 2**oversampling_fr)
        N_input_fr = (self.J+1) * self.Q[0]
        K_fr = max(self.J_fr - self.oversampling_fr, 0)
        N_padded_fr_subsampled = (N_input_fr + min_to_pad_fr) // (2 ** K_fr)
        self._N_padded_fr = N_padded_fr_subsampled * (2 ** K_fr)

    def create_filters(self):
        phi0_fr_f,= scattering_filter_factory(self._N_padded_fr,
            self.J_fr, (), self.F, self.filterbank_fr, _reduction=np.sum)
        phi1_fr_f, psis_fr_f = scattering_filter_factory(self._N_padded_fr,
            self.J_fr, self.Q_fr, 2**self.J_fr, self.filterbank_fr,
            _reduction=np.sum)
        self.filters_fr = (phi0_fr_f, [phi1_fr_f] + psis_fr_f)

        # Check for absence of aliasing
        assert all((abs(psi1["xi"]) < 0.5/(2**psi1["j"])) for psi1 in psis_fr_f)

    def scattering(self, x):
        TimeFrequencyScatteringBase._check_runtime_args(self)
        TimeFrequencyScatteringBase._check_input(self, x)

        x_shape = self.backend.shape(x)
        batch_shape, signal_shape = x_shape[:-1], x_shape[-1:]
        x = self.backend.reshape_input(x, signal_shape)
        U_0 = self.backend.pad(
            x, pad_left=self.pad_left, pad_right=self.pad_right)

        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        U_gen = joint_timefrequency_scattering(U_0, self.backend,
            filters, self.log2_stride, (self.average=='local'),
            self.filters_fr, self.log2_stride_fr, (self.average_fr=='local'))

        S_gen = jtfs_average_and_format(U_gen, self.backend,
            self.phi_f, self.log2_stride, self.average,
            self.filters_fr[0], self.log2_stride_fr, self.average_fr,
            self.out_type, self.format)

        # Zeroth order
        path = next(S_gen)
        if not self.average == 'global':
            res = self.log2_stride if self.average else 0
            path['coef'] = self.backend.unpad(
                path['coef'], self.ind_start[res], self.ind_end[res])
        path['coef'] = self.backend.reshape_output(
            path['coef'], batch_shape, n_kept_dims=1)
        S = [path]

        # First and second order
        for path in S_gen:
            # Temporal unpadding. Switch cases:
            # 1. If averaging is global, no need for unpadding at all.
            # 2. If averaging is local, unpad at resolution log2_T
            # 3. If there is no averaging, unpadding depends on order:
            #     3a. at order 1, unpad Y_1_fr at resolution log2_T
            #     3b. at order 2, unpad Y_2_fr at resolution j2
            if not self.average == 'global':
                if not self.average and len(path['n']) > 1:
                    # Case 3b.
                    res = max(path['j'][-1], 0)
                else:
                    # Cases 2a, 2b, and 3a.
                    res = max(self.log2_stride, 0)
                # Cases 2a, 2b, 3a, and 3b.
                path['coef'] = self.backend.unpad(
                    path['coef'], self.ind_start[res], self.ind_end[res])

            # Reshape path to batch shape.
            path['coef'] = self.backend.reshape_output(path['coef'],
                batch_shape, n_kept_dims=(1 + (self.format == "joint")))
            S.append(path)

        if (self.format == 'joint') and (self.out_type == 'array'):
            # Skip zeroth order
            S = S[1:]
            # Stack first and second orders into a 4D tensor:
            # (batch, n_jtfs, freq, time) where n_jtfs aggregates (n2, n_fr)
            return self.backend.stack([path['coef'] for path in S], dim=-3)
        elif (self.format == "time") and (self.out_type == "array"):
            # Stack zeroth, first, and second orders into a 3D tensor:
            # (batch, n_jtfs, time) where n_jtfs aggregates (n1, n2, n_fr)
            return self.backend.stack([path['coef'] for path in S], dim=-2)
        elif self.out_type == 'dict':
            return {path['n']: path['coef'] for path in S}
        elif self.out_type == 'list':
            return S

    def meta(self):
        filters = [self.phi_f, self.psi1_f, self.psi2_f]
        U_gen = joint_timefrequency_scattering(None, self._DryBackend(),
            filters, self.log2_stride, self.average=='local',
            self.filters_fr, self.log2_stride_fr, self.average_fr=='local')
        S_gen = jtfs_average_and_format(U_gen, self._DryBackend(),
            self.phi_f, self.log2_stride, self.average,
            self.filters_fr[0], self.log2_stride_fr, self.average_fr,
            self.out_type, self.format)
        S = sorted(list(S_gen), key=lambda path: (len(path['n']), path['n']))
        meta = dict(key=[path['n'] for path in S], n=[], n_fr=[], order=[])
        for path in S:
            if len(path['n']) == 0:
                # If format='joint' and out_type='array' skip zeroth order
                if not (self.format == 'joint' and self.out_type == 'array'):
                    # Zeroth order: no n1, no n_fr, no n2
                    meta['n'].append([np.nan, np.nan])
                    meta['n_fr'].append(np.nan)
                    meta['order'].append(0)
            else:
                if len(path['n']) == 1:
                    # First order and format='joint': n=(n_fr,)
                    n1_range = range(0, path['n1_max'], path['n1_stride'])
                    meta['n'].append([n1_range, np.nan])
                elif len(path['n']) == 2 and self.format == 'joint':
                    # Second order and format='joint': n=(n2, n_fr)
                    n1_range = range(0, path['n1_max'], path['n1_stride'])
                    meta['n'].append([n1_range, path['n'][0]])
                elif len(path['n']) == 2 and self.format == 'time':
                    # First order and format='time': n=(n1, n_fr)
                    meta['n'].append([path['n'][0], np.nan])
                elif len(path['n']) == 3 and self.format == 'time':
                    # Second order and format='time': n=(n1, n2, n_fr)
                    meta['n'].append(path['n'][:2])
                meta['n_fr'].append(path['n_fr'][0])
                meta['order'].append(len(path['n']) - (self.format == 'time'))
        meta['n'] = np.array(meta['n'], dtype=object)
        meta['n_fr'] = np.array(meta['n_fr'])
        meta['order'] = np.array(meta['order'])
        for key in ['xi', 'sigma', 'j']:
            meta[key] = np.zeros((meta['n_fr'].shape[0], 2)) * np.nan
            for order, filterbank in enumerate(filters[1:]):
                for n, psi in enumerate(filterbank):
                    meta[key][meta['n'][:, order]==n, order] = psi[key]
            meta[key + '_fr'] = meta['n_fr'] * np.nan
            for n_fr, psi_fr in enumerate(self.filters_fr[1]):
                meta[key + '_fr'][meta['n_fr']==n_fr] = psi_fr[key]
        meta['spin'] = np.sign(meta['xi_fr'])
        return meta

    def _check_runtime_args(self):
        super(TimeFrequencyScatteringBase, self)._check_runtime_args()

        if self.format == 'joint':
            if (not self.average_fr) and (self.out_type == 'array'):
                raise ValueError("Cannot convert to format='joint' with "
                "out_type='array' and F=0. Either set format='time', "
                "out_type='dict', or out_type='list'.")

        if self.oversampling_fr < 0:
            raise ValueError("oversampling_fr must be nonnegative. Got: {}".format(
                self.oversampling_fr))

        if not isinstance(self.oversampling_fr, numbers.Integral):
            raise ValueError("oversampling_fr must be integer. Got: {}".format(
                self.oversampling_fr))

        if self.format not in ['time', 'joint']:
            raise ValueError("format must be 'time' or 'joint'. Got: {}".format(
                self.format))

    @property
    def average_fr(self):
        N_input_fr = (self.J+1) * self.Q[0]
        return parse_T(self._F, self.J_fr, N_input_fr, T_alias='F')[1]

    @property
    def F(self):
        N_input_fr = (self.J+1) * self.Q[0]
        return parse_T(self._F, self.J_fr, N_input_fr, T_alias='F')[0]

    @property
    def filterbank_fr(self):
        filterbank_kwargs = {
            "alpha": self.alpha, "r_psi": self.r_psi, "sigma0": self.sigma0}
        return spin(anden_generator, filterbank_kwargs)

    @property
    def log2_F(self):
        return int(math.floor(math.log2(self.F)))

    @property
    def log2_stride_fr(self):
        if self._stride_fr is None:
            return self.log2_F
        if not isinstance(self._stride_fr, numbers.Integral):
            raise ValueError("stride_fr must be integer. Got: {}".format(
                self._stride_fr))
        log2_stride_fr = math.log2(self._stride_fr)
        if math.floor(log2_stride_fr) != math.ceil(log2_stride_fr):
            raise ValueError("stride_fr must be a power of two. Got: {}".format(
                self._stride_fr))
        if self.average_fr in [False, "global"]:
            raise ValueError("stride_fr={} is incompatible with F={}.".format(
                self._stride, self._F))
        return int(log2_stride_fr)

    @property
    def stride_fr(self):
        return (2**self.log2_stride_fr)

    _doc_instantiation_shape = {True:
                                'S = TimeFrequencyScattering(J=J_time, J_fr=J_freq, Q=Q, shape=N)',
                                False:
                                'S = TimeFrequencyScattering(J=J_time, J_fr=J_freq, Q=Q)'}

    _doc_class = \
    r"""The joint time-frequency scattering transform (JTFS)

        The joint time-frequency scattering transform first decomposes a
        signal in frequency using a wavelet filter bank, then computes a
        second wavelet decomposition in this time-frequency domain using a
        2D wavelet transform. In particular, given a 1D signal $x(t)$,
        we define the scalogram

            $X(t, \lambda) = |x \star \psi_\lambda(t)|,$

        where $\star$ denotes convolution and $\psi_\lambda(t)$ is a wavelet
        centered at frequency $2^\lambda$, that is, centered at log-frequency
        $\lambda$. The scattering transform may now be written as

            $S_J x = [S_J^{{(0)}} x, S_J^{{(1)}} x, S_J^{{(2)}} x]$

        where

            $S_J^{{(0)}} x(t) = x \star \phi_J(t)$,

            $S_J^{{(1)}} x(t, \lambda) = X \star \Phi_{{T,F}}(t, \lambda)$

            $S_J^{{(2)}} x(t, \lambda, \mu, \ell, s) = |X \star \Psi_{{\mu,
            \ell, s}}| \star \Phi_{{T,F}}(t, \lambda)$

        Here, $\Phi_{{T,F}}(t, \lambda)$ is a lowpass filter with extent $T$
        in time and $F$ in log-frequency, while $\Psi_{{\mu, \ell, s}}(t,
        \lambda)$ is a two-dimensional wavelet with time frequency $2^\mu$,
        quefrency $2^\ell$ and direction $s$ ($1$ for up, $-1$ for down).

        The `TimeFrequencyScattering` class implements the joint
        time-frequency scattering transform for a given set of filters whose
        parameters are specified at initialization.{frontend_paragraph}

        Given an input `{array}` `x` of shape `(B, N)`, where `B` is the
        number of signals to transform (the batch size) and `N` is the length
        of the signal, we compute its scattering transform by passing it to
        the `scattering` method (or calling the alias `{alias_name}`). Note
        that `B` can be one, in which case it may be omitted, giving an input
        of shape `(N,)`.

        Example
        -------
        ::

            # Set the parameters of the scattering transform.
            J_time = 6
            J_freq = 2
            N = 2 ** 13
            Q = 8

            # Generate a sample signal.
            x = {sample}

            # Define a TimeFrequencyScattering object.
            {instantiation}

            # Calculate the scattering transform.
            Sx = S.scattering(x)

            # Equivalently, use the alias.
            Sx = S{alias_call}

        Above, the length of the signal is $N = 2^{{13}} = 8192$, while the
        maximum time scale of the transform is set to
        $2^{{J_{{\text{{time}}}}}} = 2^6 = 64$ and the maximum frequency scale
        is set to $2^{{J_{{\text{{freq}}}}}} = 2^2 = 4$. The time-frequency
        resolution of the first-order wavelets :math:`\psi_\lambda(t)` is set
        to `Q = 8` wavelets per octave. The second-order wavelets $\Psi_{{\mu,
        \ell, s}}(t, \lambda)$ default to one wavelet per octave in both the
        time and log-frequency axes.

        Parameters
        ----------
        J : int
            The maximum log-scale in time of the scattering transform. In
            other words, the maximum time scale is given by $2^J$.
        J_fr : int
            The maximum log-scale in log-frequency of the scattering
            transform. In other words, the maximum log-frequency scale is
            given by $2^{{J_\text{{fr}}}}$, measured in number of filters, where
            $Q$ filters (see below) make up an octave.
        Q : int or tuple
            By default, Q (int) is the number of wavelets per octave for the first
            order and second order has one wavelet per octave in time. This
            default value can be modified by passing Q as a tuple with two values,
            i.e. `Q = (Q1, Q2)`, where `Q1` and `Q2` are the number of wavelets per
            octave for the first and second order, respectively.
        T : int
            The temporal support of low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling.
        stride : int
            The stride with which the scattering transform is sampled.
            When set to `1`, no subsampling is performed. Must be a power of
            two. Defaults to `2 ** J`.
        Q_fr : int
            The number of wavelets per octave in frequency for the
            second-order wavelets. Defaults to `1`.
        F : int
            The log-frequency support of low-pass filter, controlling amount
            of imposed transposition invariance and maximum subsampling.
            Measured in number of filters, where we have `Q1 filtersper octave
            (see above).
        stride_fr : int
            The stride with which the scattering transform is sampled along
            the log-frequency axis. When set to `1`, no subsampling is
            performed. Must be a power of two. Defaults to `2 ** J_fr`.{param_vectorize}
        format : str
            Either `time` (default) or `joint`. In the former case, all
            coefficient are mixed along the channel dimension, aggregating the
            first-order filter index, the second-order time filter index, and
            the second order frequency filter index. For the `joint` format,
            the first-order filter index is separated out, yielding a stack of
            time-frequency images indexed by the second-order time filter
            index and the second-order frequency filter index.

        Attributes
        ----------
        J : int
            The maximum log-scale of the transform in time. In other words,
            the maximum scale is given by `2 ** J`.
        J_fr : int
            The maximum log-scale of the transform in log-frequency. In other
            words, the maximum scale is given by `2 ** J_fr`.
        Q : int
            The number of first-order wavelets per octave (second-order
            wavelets are fixed to one wavelet per octave).{param_shape}
        T : int
            Temporal support of low-pass filter, controlling amount of imposed
            time-shift invariance and maximum subsampling.
        F : int
            Log-frequency support of low-pass filter, controlling amount of
            imposed transposition invariance and maximum
            subsampling.{attrs_shape}{attr_vectorize}
"""

    _doc_scattering = \
    """Apply the scattering transform

       Given an input `{array}` of size `(B, N)`, where `B` is the batch
       size (it can be potentially an integer or a shape) and `N` is the length
       of the individual signals, this function computes its scattering
       transform. If the `out_type` is set to `'array'` and `format` is set to
       `'time'`, the output is in the form of a `{array}` of size `(B, C, N1)`,
       where `N1` is the signal length after subsampling to the scale :math:`2^J`
       (with the appropriate oversampling factor to reduce aliasing), and `C` is
       the number of scattering coefficients. If `format` is set to `'joint',
       the shape will be `(B, C, L1, N1)`, where `L1` is the number of
       frequency indices after subsampling and `C` is the number of
       second-order scattering coefficients. If `out_type` is set to `'list'`,
       however, the output is a list of dictionaries, each dictionary
       corresponding to a scattering coefficient and its associated meta
       information. The coefficient is stored under the `'coef'` key, while
       other keys contain additional information, such as `'j'` (the scale of
       the filter used) and `'n`' (the filter index).

       Parameters
       ----------
       x : {array}
           An input `{array}` of size `(B, N)`.

       Returns
       -------
       S : tensor or list
           If `out_type` is `'array'` the output is a{n} `{array}` containing
           the scattering coefficients, while if `out_type` is `'list'`, the
           output is a list of dictionaries as described above."""

