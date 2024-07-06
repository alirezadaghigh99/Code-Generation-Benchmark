def from_dict_of_dict(
        cls,
        dict_of_dict,
        n_tracks,
        min_length=10,
        split_tracklets=True,
        prestitch_residuals=True,
    ):
        tracklets = []
        header = dict_of_dict.pop("header", None)
        single = None
        for k, dict_ in dict_of_dict.items():
            try:
                inds, data = zip(*[(cls.get_frame_ind(k), v) for k, v in dict_.items()])
            except ValueError:
                continue
            inds = np.asarray(inds)
            data = np.asarray(data)
            try:
                nrows, ncols = data.shape
                data = data.reshape((nrows, ncols // 3, 3))
            except ValueError:
                pass
            tracklet = Tracklet(data, inds)
            if k == "single":
                single = tracklet
            else:
                tracklets.append(Tracklet(data, inds))
        class_ = cls(
            tracklets, n_tracks, min_length, split_tracklets, prestitch_residuals
        )
        class_.header = header
        class_.single = single
        return class_

