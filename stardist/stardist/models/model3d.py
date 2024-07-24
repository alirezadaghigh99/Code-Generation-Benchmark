class StarDist3D(StarDistBase):
    """StarDist3D model.

    Parameters
    ----------
    config : :class:`Config` or None
        Will be saved to disk as JSON (``config.json``).
        If set to ``None``, will be loaded from disk (must exist).
    name : str or None
        Model name. Uses a timestamp if set to ``None`` (default).
    basedir : str
        Directory that contains (or will contain) a folder with the given model name.

    Raises
    ------
    FileNotFoundError
        If ``config=None`` and config cannot be loaded from disk.
    ValueError
        Illegal arguments, including invalid configuration.

    Attributes
    ----------
    config : :class:`Config`
        Configuration, as provided during instantiation.
    keras_model : `Keras model <https://keras.io/getting-started/functional-api-guide/>`_
        Keras neural network model.
    name : str
        Model name.
    logdir : :class:`pathlib.Path`
        Path to model folder (which stores configuration, weights, etc.)
    """

    def __init__(self, config=Config3D(), name=None, basedir='.'):
        """See class docstring."""
        super().__init__(config, name=name, basedir=basedir)


    def _build(self):
        if self.config.backbone == "unet":
            return self._build_unet()
        elif self.config.backbone == "resnet":
            return self._build_resnet()
        else:
            raise NotImplementedError(self.config.backbone)


    def _build_unet(self):
        assert self.config.backbone == 'unet'
        unet_kwargs = {k[len('unet_'):]:v for (k,v) in vars(self.config).items() if k.startswith('unet_')}

        input_img = Input(self.config.net_input_shape, name='input')

        # maxpool input image to grid size
        pooled = np.array([1,1,1])
        pooled_img = input_img
        while tuple(pooled) != tuple(self.config.grid):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            for _ in range(self.config.unet_n_conv_per_depth):
                pooled_img = Conv3D(self.config.unet_n_filter_base, self.config.unet_kernel_size,
                                    padding='same', activation=self.config.unet_activation)(pooled_img)
            pooled_img = MaxPooling3D(pool)(pooled_img)

        unet_base = unet_block(**unet_kwargs)(pooled_img)

        if self.config.net_conv_after_unet > 0:
            unet = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                          name='features', padding='same', activation=self.config.unet_activation)(unet_base)
        else:
            unet = unet_base

        output_prob = Conv3D(                 1, (1,1,1), name='prob', padding='same', activation='sigmoid')(unet)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='linear')(unet)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_unet > 0:
                unet_class  = Conv3D(self.config.net_conv_after_unet, self.config.unet_kernel_size,
                                     name='features_class', padding='same', activation=self.config.unet_activation)(unet_base)
            else:
                unet_class  = unet_base

            output_prob_class  = Conv3D(self.config.n_classes+1, (1,1,1), name='prob_class', padding='same', activation='softmax')(unet_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])


    def _build_resnet(self):
        assert self.config.backbone == 'resnet'
        n_filter = self.config.resnet_n_filter_base
        resnet_kwargs = dict (
            kernel_size        = self.config.resnet_kernel_size,
            n_conv_per_block   = self.config.resnet_n_conv_per_block,
            batch_norm         = self.config.resnet_batch_norm,
            kernel_initializer = self.config.resnet_kernel_init,
            activation         = self.config.resnet_activation,
        )

        input_img = Input(self.config.net_input_shape, name='input')

        layer = input_img
        layer = Conv3D(n_filter, (7,7,7), padding="same", kernel_initializer=self.config.resnet_kernel_init)(layer)
        layer = Conv3D(n_filter, (3,3,3), padding="same", kernel_initializer=self.config.resnet_kernel_init)(layer)

        pooled = np.array([1,1,1])
        for n in range(self.config.resnet_n_blocks):
            pool = 1 + (np.asarray(self.config.grid) > pooled)
            pooled *= pool
            if any(p > 1 for p in pool):
                n_filter *= 2
            layer = resnet_block(n_filter, pool=tuple(pool), **resnet_kwargs)(layer)

        layer_base = layer

        if self.config.net_conv_after_resnet > 0:
            layer = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                           name='features', padding='same', activation=self.config.resnet_activation)(layer_base)

        output_prob = Conv3D(                 1, (1,1,1), name='prob', padding='same', activation='sigmoid')(layer)
        output_dist = Conv3D(self.config.n_rays, (1,1,1), name='dist', padding='same', activation='linear')(layer)

        # attach extra classification head when self.n_classes is given
        if self._is_multiclass():
            if self.config.net_conv_after_resnet > 0:
                layer_class  = Conv3D(self.config.net_conv_after_resnet, self.config.resnet_kernel_size,
                                      name='features_class', padding='same', activation=self.config.resnet_activation)(layer_base)
            else:
                layer_class  = layer_base

            output_prob_class  = Conv3D(self.config.n_classes+1, (1,1,1), name='prob_class', padding='same', activation='softmax')(layer_class)
            return Model([input_img], [output_prob,output_dist,output_prob_class])
        else:
            return Model([input_img], [output_prob,output_dist])


    def train(self, X, Y, validation_data, classes='auto', augmenter=None, seed=None, epochs=None, steps_per_epoch=None, workers=1):
        """Train the neural network with the given data.

        Parameters
        ----------
        X : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Input images
        Y : tuple, list, `numpy.ndarray`, `keras.utils.Sequence`
            Label masks
            Positive pixel values denote object instance ids (0 for background).
            Negative values can be used to turn off all losses for the corresponding pixels (e.g. for regions that haven't been labeled).
        classes (optional): 'auto' or iterable of same length as X
             label id -> class id mapping for each label mask of Y if multiclass prediction is activated (n_classes > 0)
             list of dicts with label id -> class id (1,...,n_classes)
             'auto' -> all objects will be assigned to the first non-background class,
                       or will be ignored if config.n_classes is None
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`) or triple (if multiclass)
            Tuple (triple if multiclass) of X,Y,[classes] validation data.
        augmenter : None or callable
            Function with expected signature ``xt, yt = augmenter(x, y)``
            that takes in a single pair of input/label image (x,y) and returns
            the transformed images (xt, yt) for the purpose of data augmentation
            during training. Not applied to validation images.
            Example:
            def simple_augmenter(x,y):
                x = x + 0.05*np.random.normal(0,1,x.shape)
                return x,y
        seed : int
            Convenience to set ``np.random.seed(seed)``. (To obtain reproducible validation patches, etc.)
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.

        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.

        """
        if seed is not None:
            # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
            np.random.seed(seed)
        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        classes = self._parse_classes_arg(classes, len(X))

        if not self._is_multiclass() and classes is not None:
            warnings.warn("Ignoring given classes as n_classes is set to None")

        isinstance(validation_data,(list,tuple)) or _raise(ValueError())
        if self._is_multiclass() and len(validation_data) == 2:
            validation_data = tuple(validation_data) + ('auto',)
        ((len(validation_data) == (3 if self._is_multiclass() else 2))
            or _raise(ValueError(f'len(validation_data) = {len(validation_data)}, but should be {3 if self._is_multiclass() else 2}')))

        patch_size = self.config.train_patch_size
        axes = self.config.axes.replace('C','')
        div_by = self._axes_div_by(axes)
        [p % d == 0 or _raise(ValueError(
            "'train_patch_size' must be divisible by {d} along axis '{a}'".format(a=a,d=d)
         )) for p,d,a in zip(patch_size,div_by,axes)]

        if not self._model_prepared:
            self.prepare_for_training()

        data_kwargs = dict (
            rays             = rays_from_json(self.config.rays_json),
            grid             = self.config.grid,
            patch_size       = self.config.train_patch_size,
            anisotropy       = self.config.anisotropy,
            use_gpu          = self.config.use_gpu,
            foreground_prob  = self.config.train_foreground_only,
            n_classes        = self.config.n_classes,
            sample_ind_cache = self.config.train_sample_cache,
        )
        worker_kwargs = dict(workers=workers, use_multiprocessing=workers>1)
        if IS_KERAS_3_PLUS:
            data_kwargs['keras_kwargs'] = worker_kwargs
            fit_kwargs = {}
        else:
            fit_kwargs = worker_kwargs

        # generate validation data and store in numpy arrays
        n_data_val = len(validation_data[0])
        classes_val = self._parse_classes_arg(validation_data[2], n_data_val) if self._is_multiclass() else None
        n_take = self.config.train_n_val_patches if self.config.train_n_val_patches is not None else n_data_val
        _data_val = StarDistData3D(validation_data[0],validation_data[1], classes=classes_val, batch_size=n_take, length=1, **data_kwargs)
        data_val = _data_val[0]

        # expose data generator as member for general diagnostics
        self.data_train = StarDistData3D(X, Y, classes=classes, batch_size=self.config.train_batch_size,
                                         augmenter=augmenter, length=epochs*steps_per_epoch, **data_kwargs)

        if self.config.train_tensorboard:
            # only show middle slice of 3D inputs/outputs
            input_slices, output_slices = [[slice(None)]*5], [[slice(None)]*5,[slice(None)]*5]
            i = axes_dict(self.config.axes)['Z']
            channel = axes_dict(self.config.axes)['C']
            _n_in  = _data_val.patch_size[i] // 2
            _n_out = _data_val.patch_size[i] // (2 * (self.config.grid[i] if self.config.grid is not None else 1))
            input_slices[0][1+i] = _n_in
            output_slices[0][1+i] = _n_out
            output_slices[1][1+i] = _n_out
            # show dist for three rays
            _n = min(3, self.config.n_rays)
            output_slices[1][1+channel] = slice(0,(self.config.n_rays//_n)*_n, self.config.n_rays//_n)
            if self._is_multiclass():
                _n = min(3, self.config.n_classes)
                output_slices += [[slice(None)]*5]
                output_slices[2][1+channel] = slice(1,1+(self.config.n_classes//_n)*_n, self.config.n_classes//_n)

            if IS_TF_1:
                for cb in self.callbacks:
                    if isinstance(cb,CARETensorBoard):
                        cb.input_slices = input_slices
                        cb.output_slices = output_slices
                        # target image for dist includes dist_mask and thus has more channels than dist output
                        cb.output_target_shapes = [None,[None]*5,None]
                        cb.output_target_shapes[1][1+channel] = data_val[1][1].shape[1+channel]
            elif self.basedir is not None and not any(isinstance(cb,CARETensorBoardImage) for cb in self.callbacks):
                self.callbacks.append(CARETensorBoardImage(model=self.keras_model, data=data_val, log_dir=str(self.logdir/'logs'/'images'),
                                                           n_images=3, prob_out=False, input_slices=input_slices, output_slices=output_slices))

        fit = self.keras_model.fit_generator if (IS_TF_1 and not IS_KERAS_3_PLUS) else self.keras_model.fit
        history = fit(iter(self.data_train), validation_data=data_val,
                      epochs=epochs, steps_per_epoch=steps_per_epoch,
                      **fit_kwargs,
                      callbacks=self.callbacks, verbose=1,
                      # set validation batchsize to training batchsize (only works in tf 2.x)
                      **(dict(validation_batch_size = self.config.train_batch_size) if _tf_version_at_least("2.2.0") else {}))
        self._training_finished()

        return history


    def _instances_from_prediction(self, img_shape, prob, dist, points=None, prob_class=None, prob_thresh=None, nms_thresh=None, overlap_label=None, return_labels=True, scale=None, **nms_kwargs):
        """
        if points is None     -> dense prediction
        if points is not None -> sparse prediction

        if prob_class is None     -> single class prediction
        if prob_class is not None -> multi  class prediction
        """
        if prob_thresh is None: prob_thresh = self.thresholds.prob
        if nms_thresh  is None: nms_thresh  = self.thresholds.nms

        rays = rays_from_json(self.config.rays_json)

        # sparse prediction
        if points is not None:
            points, probi, disti, indsi = non_maximum_suppression_3d_sparse(dist, prob, points, rays, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                prob_class = prob_class[indsi]

        # dense prediction
        else:
            points, probi, disti = non_maximum_suppression_3d(dist, prob, rays, grid=self.config.grid,
                                                              prob_thresh=prob_thresh, nms_thresh=nms_thresh, **nms_kwargs)
            if prob_class is not None:
                inds = tuple(p//g for p,g in zip(points.T, self.config.grid))
                prob_class = prob_class[inds]

        verbose = nms_kwargs.get('verbose',False)
        verbose and print("render polygons...")

        if scale is not None:
            # need to undo the scaling given by the scale dict, e.g. scale = dict(X=0.5,Y=0.5,Z=1.0):
            #   1. re-scale points (origins of polyhedra)
            #   2. re-scale vectors of rays object (computed from distances)
            if not (isinstance(scale,dict) and 'X' in scale and 'Y' in scale and 'Z' in scale):
                raise ValueError("scale must be a dictionary with entries for 'X', 'Y', and 'Z'")
            rescale = (1/scale['Z'],1/scale['Y'],1/scale['X'])
            points = points * np.array(rescale).reshape(1,3)
            rays = rays.copy(scale=rescale)
        else:
            rescale = (1,1,1)

        if return_labels:
            labels = polyhedron_to_label(disti, points, rays=rays, prob=probi, shape=img_shape, overlap_label=overlap_label, verbose=verbose)

            # map the overlap_label to something positive and back
            # (as relabel_sequential doesn't like negative values)
            if overlap_label is not None and overlap_label<0 and (overlap_label in labels):
                overlap_mask = (labels == overlap_label)
                overlap_label2 = max(set(np.unique(labels))-{overlap_label})+1
                labels[overlap_mask] = overlap_label2
                labels, fwd, bwd = relabel_sequential(labels)
                labels[labels == fwd[overlap_label2]] = overlap_label
            else:
                # TODO relabel_sequential necessary?
                # print(np.unique(labels))
                labels, _,_ = relabel_sequential(labels)
                # print(np.unique(labels))
        else:
            labels = None

        res_dict = dict(dist=disti, points=points, prob=probi, rays=rays, rays_vertices=rays.vertices, rays_faces=rays.faces)

        if prob_class is not None:
            # build the list of class ids per label via majority vote
            # zoom prob_class to img_shape
            # prob_class_up = zoom(prob_class,
            #                      tuple(s2/s1 for s1, s2 in zip(prob_class.shape[:3], img_shape))+(1,),
            #                      order=0)
            # class_id, label_ids = [], []
            # for reg in regionprops(labels):
            #     m = labels[reg.slice]==reg.label
            #     cls_id = np.argmax(np.mean(prob_class_up[reg.slice][m], axis = 0))
            #     class_id.append(cls_id)
            #     label_ids.append(reg.label)
            # # just a sanity check whether labels where in sorted order
            # assert all(x <= y for x,y in zip(label_ids, label_ids[1:]))
            # res_dict.update(dict(classes = class_id))
            # res_dict.update(dict(labels = label_ids))
            # self.p = prob_class_up

            prob_class = np.asarray(prob_class)
            class_id = np.argmax(prob_class, axis=-1)
            res_dict.update(dict(class_prob=prob_class, class_id=class_id))

        return labels, res_dict


    def _axes_div_by(self, query_axes):
        if self.config.backbone == "unet":
            query_axes = axes_check_and_normalize(query_axes)
            assert len(self.config.unet_pool) == len(self.config.grid)
            div_by = dict(zip(
                self.config.axes.replace('C',''),
                tuple(p**self.config.unet_n_depth * g for p,g in zip(self.config.unet_pool,self.config.grid))
            ))
            return tuple(div_by.get(a,1) for a in query_axes)
        elif self.config.backbone == "resnet":
            grid_dict = dict(zip(self.config.axes.replace('C',''), self.config.grid))
            return tuple(grid_dict.get(a,1) for a in query_axes)
        else:
            raise NotImplementedError()


    @property
    def _config_class(self):
        return Config3D

