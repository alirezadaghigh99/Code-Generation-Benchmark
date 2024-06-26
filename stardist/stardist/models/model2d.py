class StarDistData2D(StarDistDataBase):

    def __init__(self, X, Y, batch_size, n_rays, length,
                 n_classes=None, classes=None,
                 patch_size=(256,256), b=32, grid=(1,1), shape_completion=False, augmenter=None, foreground_prob=0, **kwargs):

        super().__init__(X=X, Y=Y, n_rays=n_rays, grid=grid,
                         n_classes=n_classes, classes=classes,
                         batch_size=batch_size, patch_size=patch_size, length=length,
                         augmenter=augmenter, foreground_prob=foreground_prob, **kwargs)

        self.shape_completion = bool(shape_completion)
        if self.shape_completion and b > 0:
            if not all(b % g == 0 for g in self.grid):
                raise ValueError(f"'shape_completion' requires that crop size {b} ('train_completion_crop' in config) is evenly divisible by all grid values {self.grid}")
            self.b = slice(b,-b),slice(b,-b)
        else:
            self.b = slice(None),slice(None)

        self.sd_mode = 'opencl' if self.use_gpu else 'cpp'


    def __getitem__(self, i):
        idx = self.batch(i)
        arrays = [sample_patches((self.Y[k],) + self.channels_as_tuple(self.X[k]),
                                 patch_size=self.patch_size, n_samples=1,
                                 valid_inds=self.get_valid_inds(k)) for k in idx]

        if self.n_channel is None:
            X, Y = list(zip(*[(x[0][self.b],y[0]) for y,x in arrays]))
        else:
            X, Y = list(zip(*[(np.stack([_x[0] for _x in x],axis=-1)[self.b], y[0]) for y,*x in arrays]))

        X, Y = tuple(zip(*tuple(self.augmenter(_x, _y) for _x, _y in zip(X,Y))))

        mask_neg_labels = tuple(y[self.b][self.ss_grid[1:3]] < 0 for y in Y)
        has_neg_labels = any(m.any() for m in mask_neg_labels)
        if has_neg_labels:
            mask_neg_labels = np.stack(mask_neg_labels)
            # set negative label pixels to 0 (background)
            Y = tuple(np.maximum(y, 0) for y in Y)

        prob = np.stack([edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in Y])
        # prob = np.stack([edt_prob(lbl[self.b]) for lbl in Y])
        # prob = prob[self.ss_grid]

        if self.shape_completion:
            Y_cleared = [clear_border(lbl) for lbl in Y]
            _dist     = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode)[self.b+(slice(None),)] for lbl in Y_cleared])
            dist      = _dist[self.ss_grid]
            dist_mask = np.stack([edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in Y_cleared])
        else:
            # directly subsample with grid
            dist      = np.stack([star_dist(lbl,self.n_rays,mode=self.sd_mode, grid=self.grid) for lbl in Y])
            dist_mask = prob

        X = np.stack(X)
        if X.ndim == 3: # input image has no channel axis
            X = np.expand_dims(X,-1)
        prob = np.expand_dims(prob,-1)
        dist_mask = np.expand_dims(dist_mask,-1)

        # subsample wth given grid
        # dist_mask = dist_mask[self.ss_grid]
        # prob      = prob[self.ss_grid]

        # append dist_mask to dist as additional channel
        # dist_and_mask = np.concatenate([dist,dist_mask],axis=-1)
        # faster than concatenate
        dist_and_mask = np.empty(dist.shape[:-1]+(self.n_rays+1,), np.float32)
        dist_and_mask[...,:-1] = dist
        dist_and_mask[...,-1:] = dist_mask

        if has_neg_labels:
            prob[mask_neg_labels] = -1  # set to -1 to disable loss

        # note: must return tuples in keras 3 (cf. https://stackoverflow.com/a/78158487)
        if self.n_classes is None:
            return _gen_rtype((X,)), _gen_rtype((prob,dist_and_mask))
        else:
            prob_class = np.stack(tuple((mask_to_categorical(y[self.b], self.n_classes, self.classes[k]) for y,k in zip(Y, idx))))

            # TODO: investigate downsampling via simple indexing vs. using 'zoom'
            # prob_class = prob_class[self.ss_grid]
            # 'zoom' might lead to better registered maps (especially if upscaled later)
            prob_class = zoom(prob_class, (1,)+tuple(1/g for g in self.grid)+(1,), order=0)

            if has_neg_labels:
                prob_class[mask_neg_labels] = -1  # set to -1 to disable loss

            return _gen_rtype((X,)), _gen_rtype((prob,dist_and_mask, prob_class))