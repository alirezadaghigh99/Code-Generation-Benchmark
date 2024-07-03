    def export_TF(self, fname=None, single_output=True, upsample_grid=True):
        """Export model to TensorFlow's SavedModel format that can be used e.g. in the Fiji plugin

        Parameters
        ----------
        fname : str
            Path of the zip file to store the model
            If None, the default path "<modeldir>/TF_SavedModel.zip" is used
        single_output: bool
            If set, concatenates the two model outputs into a single output (note: this is currently mandatory for further use in Fiji)
        upsample_grid: bool
            If set, upsamples the output to the input shape (note: this is currently mandatory for further use in Fiji)
        """
        Concatenate, UpSampling2D, UpSampling3D, Conv2DTranspose, Conv3DTranspose = keras_import('layers', 'Concatenate', 'UpSampling2D', 'UpSampling3D', 'Conv2DTranspose', 'Conv3DTranspose')
        Model = keras_import('models', 'Model')

        if self.basedir is None and fname is None:
            raise ValueError("Need explicit 'fname', since model directory not available (basedir=None).")

        if self._is_multiclass():
            warnings.warn("multi-class mode not supported yet, removing classification output from exported model")

        grid = self.config.grid
        prob = self.keras_model.outputs[0]
        dist = self.keras_model.outputs[1]
        assert self.config.n_dim in (2,3)

        if upsample_grid and any(g>1 for g in grid):
            # CSBDeep Fiji plugin needs same size input/output
            # -> we need to upsample the outputs if grid > (1,1)
            # note: upsampling prob with a transposed convolution creates sparse
            #       prob output with less candidates than with standard upsampling
            conv_transpose = Conv2DTranspose if self.config.n_dim==2 else Conv3DTranspose
            upsampling     = UpSampling2D    if self.config.n_dim==2 else UpSampling3D
            prob = conv_transpose(1, (1,)*self.config.n_dim,
                                  strides=grid, padding='same',
                                  kernel_initializer='ones', use_bias=False)(prob)
            dist = upsampling(grid)(dist)

        inputs  = self.keras_model.inputs[0]
        outputs = Concatenate()([prob,dist]) if single_output else [prob,dist]
        csbdeep_model = Model(inputs, outputs)

        fname = (self.logdir / 'TF_SavedModel.zip') if fname is None else Path(fname)
        export_SavedModel(csbdeep_model, str(fname))
        return csbdeep_model