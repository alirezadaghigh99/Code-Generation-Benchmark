def _parse_ground_truth_data(data):
    gt = dict()
    for i, arr in enumerate(data):
        temp = []
        for row in arr:
            if np.isnan(row[:, :2]).all():
                continue
            ass = Assembly.from_array(row)
            temp.append(ass)
        if not temp:
            continue
        gt[i] = temp
    return gt

class Assembler:
    def __init__(
        self,
        data,
        *,
        max_n_individuals,
        n_multibodyparts,
        graph=None,
        paf_inds=None,
        greedy=False,
        pcutoff=0.1,
        min_affinity=0.05,
        min_n_links=2,
        max_overlap=0.8,
        identity_only=False,
        nan_policy="little",
        force_fusion=False,
        add_discarded=False,
        window_size=0,
        method="m1",
    ):
        self.data = data
        self.metadata = self.parse_metadata(self.data)
        self.max_n_individuals = max_n_individuals
        self.n_multibodyparts = n_multibodyparts
        self.n_uniquebodyparts = self.n_keypoints - n_multibodyparts
        self.greedy = greedy
        self.pcutoff = pcutoff
        self.min_affinity = min_affinity
        self.min_n_links = min_n_links
        self.max_overlap = max_overlap
        self._has_identity = "identity" in self[0]
        if identity_only and not self._has_identity:
            warnings.warn(
                "The network was not trained with identity; setting `identity_only` to False."
            )
        self.identity_only = identity_only & self._has_identity
        self.nan_policy = nan_policy
        self.force_fusion = force_fusion
        self.add_discarded = add_discarded
        self.window_size = window_size
        self.method = method
        self.graph = graph or self.metadata["paf_graph"]
        self.paf_inds = paf_inds or self.metadata["paf"]
        self._gamma = 0.01
        self._trees = dict()
        self.safe_edge = False
        self._kde = None
        self.assemblies = dict()
        self.unique = dict()

    def __getitem__(self, item):
        return self.data[self.metadata["imnames"][item]]

    @property
    def n_keypoints(self):
        return self.metadata["num_joints"]

    def calibrate(self, train_data_file):
        df = pd.read_hdf(train_data_file)
        try:
            df.drop("single", level="individuals", axis=1, inplace=True)
        except KeyError:
            pass
        n_bpts = len(df.columns.get_level_values("bodyparts").unique())
        if n_bpts == 1:
            warnings.warn("There is only one keypoint; skipping calibration...")
            return

        xy = df.to_numpy().reshape((-1, n_bpts, 2))
        frac_valid = np.mean(~np.isnan(xy), axis=(1, 2))
        # Only keeps skeletons that are more than 90% complete
        xy = xy[frac_valid >= 0.9]
        if not xy.size:
            warnings.warn("No complete poses were found. Skipping calibration...")
            return

        # TODO Normalize dists by longest length?
        # TODO Smarter imputation technique (Bayesian? Grassmann averages?)
        dists = np.vstack([pdist(data, "sqeuclidean") for data in xy])
        mu = np.nanmean(dists, axis=0)
        missing = np.isnan(dists)
        dists = np.where(missing, mu, dists)
        try:
            kde = gaussian_kde(dists.T)
            kde.mean = mu
            self._kde = kde
            self.safe_edge = True
        except np.linalg.LinAlgError:
            # Covariance matrix estimation fails due to numerical singularities
            warnings.warn(
                "The assembler could not be robustly calibrated. Continuing without it..."
            )

    def calc_assembly_mahalanobis_dist(
        self, assembly, return_proba=False, nan_policy="little"
    ):
        if self._kde is None:
            raise ValueError("Assembler should be calibrated first with training data.")

        dists = assembly.calc_pairwise_distances() - self._kde.mean
        mask = np.isnan(dists)
        # Distance is undefined if the assembly is empty
        if not len(assembly) or mask.all():
            if return_proba:
                return np.inf, 0
            return np.inf

        if nan_policy == "little":
            inds = np.flatnonzero(~mask)
            dists = dists[inds]
            inv_cov = self._kde.inv_cov[np.ix_(inds, inds)]
            # Correct distance to account for missing observations
            factor = self._kde.d / len(inds)
        else:
            # Alternatively, reduce contribution of missing values to the Mahalanobis
            # distance to zero by substituting the corresponding means.
            dists[mask] = 0
            mask.fill(False)
            inv_cov = self._kde.inv_cov
            factor = 1
        dot = dists @ inv_cov
        mahal = factor * sqrt(np.sum((dot * dists), axis=-1))
        if return_proba:
            proba = 1 - chi2.cdf(mahal, np.sum(~mask))
            return mahal, proba
        return mahal

    def calc_link_probability(self, link):
        if self._kde is None:
            raise ValueError("Assembler should be calibrated first with training data.")

        i = link.j1.label
        j = link.j2.label
        ind = _conv_square_to_condensed_indices(i, j, self.n_multibodyparts)
        mu = self._kde.mean[ind]
        sigma = self._kde.covariance[ind, ind]
        z = (link.length ** 2 - mu) / sigma
        return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))

    @staticmethod
    def _flatten_detections(data_dict):
        ind = 0
        coordinates = data_dict["coordinates"][0]
        confidence = data_dict["confidence"]
        ids = data_dict.get("identity", None)
        if ids is None:
            ids = [np.ones(len(arr), dtype=int) * -1 for arr in confidence]
        else:
            ids = [arr.argmax(axis=1) for arr in ids]
        for i, (coords, conf, id_) in enumerate(zip(coordinates, confidence, ids)):
            if not np.any(coords):
                continue
            for xy, p, g in zip(coords, conf, id_):
                joint = Joint(tuple(xy), p.item(), i, ind, g)
                ind += 1
                yield joint

    def extract_best_links(self, joints_dict, costs, trees=None):
        links = []
        for ind in self.paf_inds:
            s, t = self.graph[ind]
            dets_s = joints_dict.get(s, None)
            dets_t = joints_dict.get(t, None)
            if dets_s is None or dets_t is None:
                continue
            if ind not in costs:
                continue
            lengths = costs[ind]["distance"]
            if np.isinf(lengths).all():
                continue
            aff = costs[ind][self.method].copy()
            aff[np.isnan(aff)] = 0

            if trees:
                vecs = np.vstack(
                    [[*det_s.pos, *det_t.pos] for det_s in dets_s for det_t in dets_t]
                )
                dists = []
                for n, tree in enumerate(trees, start=1):
                    d, _ = tree.query(vecs)
                    dists.append(np.exp(-self._gamma * n * d))
                w = np.mean(dists, axis=0)
                aff *= w.reshape(aff.shape)

            if self.greedy:
                conf = np.asarray(
                    [
                        [det_s.confidence * det_t.confidence for det_t in dets_t]
                        for det_s in dets_s
                    ]
                )
                rows, cols = np.where(
                    (conf >= self.pcutoff * self.pcutoff) & (aff >= self.min_affinity)
                )
                candidates = sorted(
                    zip(rows, cols, aff[rows, cols], lengths[rows, cols]),
                    key=lambda x: x[2],
                    reverse=True,
                )
                i_seen = set()
                j_seen = set()
                for i, j, w, l in candidates:
                    if i not in i_seen and j not in j_seen:
                        i_seen.add(i)
                        j_seen.add(j)
                        links.append(Link(dets_s[i], dets_t[j], w))
                        if len(i_seen) == self.max_n_individuals:
                            break
            else:  # Optimal keypoint pairing
                inds_s = sorted(
                    range(len(dets_s)), key=lambda x: dets_s[x].confidence, reverse=True
                )[: self.max_n_individuals]
                inds_t = sorted(
                    range(len(dets_t)), key=lambda x: dets_t[x].confidence, reverse=True
                )[: self.max_n_individuals]
                keep_s = [
                    ind for ind in inds_s if dets_s[ind].confidence >= self.pcutoff
                ]
                keep_t = [
                    ind for ind in inds_t if dets_t[ind].confidence >= self.pcutoff
                ]
                aff = aff[np.ix_(keep_s, keep_t)]
                rows, cols = linear_sum_assignment(aff, maximize=True)
                for row, col in zip(rows, cols):
                    w = aff[row, col]
                    if w >= self.min_affinity:
                        links.append(Link(dets_s[keep_s[row]], dets_t[keep_t[col]], w))
        return links

    def _fill_assembly(self, assembly, lookup, assembled, safe_edge, nan_policy):
        stack = []
        visited = set()
        tabu = []
        counter = itertools.count()

        def push_to_stack(i):
            for j, link in lookup[i].items():
                if j in assembly._idx:
                    continue
                if link.idx in visited:
                    continue
                heapq.heappush(stack, (-link.affinity, next(counter), link))
                visited.add(link.idx)

        for idx in assembly._idx:
            push_to_stack(idx)

        while stack and len(assembly) < self.n_multibodyparts:
            _, _, best = heapq.heappop(stack)
            i, j = best.idx
            if i in assembly._idx:
                new_ind = j
            elif j in assembly._idx:
                new_ind = i
            else:
                continue
            if new_ind in assembled:
                continue
            if safe_edge:
                d_old = self.calc_assembly_mahalanobis_dist(
                    assembly, nan_policy=nan_policy
                )
                success = assembly.add_link(best, store_dict=True)
                if not success:
                    assembly._dict = dict()
                    continue
                d = self.calc_assembly_mahalanobis_dist(assembly, nan_policy=nan_policy)
                if d < d_old:
                    push_to_stack(new_ind)
                    try:
                        _, _, link = heapq.heappop(tabu)
                        heapq.heappush(stack, (-link.affinity, next(counter), link))
                    except IndexError:
                        pass
                else:
                    heapq.heappush(tabu, (d - d_old, next(counter), best))
                    assembly.__dict__.update(assembly._dict)
                assembly._dict = dict()
            else:
                assembly.add_link(best)
                push_to_stack(new_ind)

    def build_assemblies(self, links):
        lookup = defaultdict(dict)
        for link in links:
            i, j = link.idx
            lookup[i][j] = link
            lookup[j][i] = link

        assemblies = []
        assembled = set()

        # Fill the subsets with unambiguous, complete individuals
        G = nx.Graph([link.idx for link in links])
        for chain in nx.connected_components(G):
            if len(chain) == self.n_multibodyparts:
                edges = [tuple(sorted(edge)) for edge in G.edges(chain)]
                assembly = Assembly(self.n_multibodyparts)
                for link in links:
                    i, j = link.idx
                    if (i, j) in edges:
                        success = assembly.add_link(link)
                        if success:
                            lookup[i].pop(j)
                            lookup[j].pop(i)
                assembled.update(assembly._idx)
                assemblies.append(assembly)

        if len(assemblies) == self.max_n_individuals:
            return assemblies, assembled

        for link in sorted(links, key=lambda x: x.affinity, reverse=True):
            if any(i in assembled for i in link.idx):
                continue
            assembly = Assembly(self.n_multibodyparts)
            assembly.add_link(link)
            self._fill_assembly(
                assembly, lookup, assembled, self.safe_edge, self.nan_policy
            )
            for link in assembly._links:
                i, j = link.idx
                lookup[i].pop(j)
                lookup[j].pop(i)
            assembled.update(assembly._idx)
            assemblies.append(assembly)

        # Fuse superfluous assemblies
        n_extra = len(assemblies) - self.max_n_individuals
        if n_extra > 0:
            if self.safe_edge:
                ds_old = [
                    self.calc_assembly_mahalanobis_dist(assembly)
                    for assembly in assemblies
                ]
                while len(assemblies) > self.max_n_individuals:
                    ds = []
                    for i, j in itertools.combinations(range(len(assemblies)), 2):
                        if assemblies[j] not in assemblies[i]:
                            temp = assemblies[i] + assemblies[j]
                            d = self.calc_assembly_mahalanobis_dist(temp)
                            delta = d - max(ds_old[i], ds_old[j])
                            ds.append((i, j, delta, d, temp))
                    if not ds:
                        break
                    min_ = sorted(ds, key=lambda x: x[2])
                    i, j, delta, d, new = min_[0]
                    if delta < 0 or len(min_) == 1:
                        assemblies[i] = new
                        assemblies.pop(j)
                        ds_old[i] = d
                        ds_old.pop(j)
                    else:
                        break
            elif self.force_fusion:
                assemblies = sorted(assemblies, key=len)
                for nrow in range(n_extra):
                    assembly = assemblies[nrow]
                    candidates = [a for a in assemblies[nrow:] if assembly not in a]
                    if not candidates:
                        continue
                    if len(candidates) == 1:
                        candidate = candidates[0]
                    else:
                        dists = []
                        for cand in candidates:
                            d = cdist(assembly.xy, cand.xy)
                            dists.append(np.nanmin(d))
                        candidate = candidates[np.argmin(dists)]
                    ind = assemblies.index(candidate)
                    assemblies[ind] += assembly
            else:
                store = dict()
                for assembly in assemblies:
                    if len(assembly) != self.n_multibodyparts:
                        for i in assembly._idx:
                            store[i] = assembly
                used = [link for assembly in assemblies for link in assembly._links]
                unconnected = [link for link in links if link not in used]
                for link in unconnected:
                    i, j = link.idx
                    try:
                        if store[j] not in store[i]:
                            temp = store[i] + store[j]
                            store[i].__dict__.update(temp.__dict__)
                            assemblies.remove(store[j])
                            for idx in store[j]._idx:
                                store[idx] = store[i]
                    except KeyError:
                        pass

        # Second pass without edge safety
        for assembly in assemblies:
            if len(assembly) != self.n_multibodyparts:
                self._fill_assembly(assembly, lookup, assembled, False, "")
                assembled.update(assembly._idx)

        return assemblies, assembled

    def _assemble(self, data_dict, ind_frame):
        joints = list(self._flatten_detections(data_dict))
        if not joints:
            return None, None

        bag = defaultdict(list)
        for joint in joints:
            bag[joint.label].append(joint)

        assembled = set()

        if self.n_uniquebodyparts:
            unique = np.full((self.n_uniquebodyparts, 3), np.nan)
            for n, ind in enumerate(range(self.n_multibodyparts, self.n_keypoints)):
                dets = bag[ind]
                if not dets:
                    continue
                if len(dets) > 1:
                    det = max(dets, key=lambda x: x.confidence)
                else:
                    det = dets[0]
                # Mark the unique body parts as assembled anyway so
                # they are not used later on to fill assemblies.
                assembled.update(d.idx for d in dets)
                if det.confidence <= self.pcutoff and not self.add_discarded:
                    continue
                unique[n] = *det.pos, det.confidence
            if np.isnan(unique).all():
                unique = None
        else:
            unique = None

        if not any(i in bag for i in range(self.n_multibodyparts)):
            return None, unique

        if self.n_multibodyparts == 1:
            assemblies = []
            for joint in bag[0]:
                if joint.confidence >= self.pcutoff:
                    ass = Assembly(self.n_multibodyparts)
                    ass.add_joint(joint)
                    assemblies.append(ass)
            return assemblies, unique

        if self.max_n_individuals == 1:
            get_attr = operator.attrgetter("confidence")
            ass = Assembly(self.n_multibodyparts)
            for ind in range(self.n_multibodyparts):
                joints = bag[ind]
                if not joints:
                    continue
                ass.add_joint(max(joints, key=get_attr))
            return [ass], unique

        if self.identity_only:
            assemblies = []
            get_attr = operator.attrgetter("group")
            temp = sorted(
                (joint for joint in joints if np.isfinite(joint.confidence)),
                key=get_attr,
            )
            groups = itertools.groupby(temp, get_attr)
            for _, group in groups:
                ass = Assembly(self.n_multibodyparts)
                for joint in sorted(group, key=lambda x: x.confidence, reverse=True):
                    if (
                        joint.confidence >= self.pcutoff
                        and joint.label < self.n_multibodyparts
                    ):
                        ass.add_joint(joint)
                if len(ass):
                    assemblies.append(ass)
                    assembled.update(ass._idx)
        else:
            trees = []
            for j in range(1, self.window_size + 1):
                tree = self._trees.get(ind_frame - j, None)
                if tree is not None:
                    trees.append(tree)

            links = self.extract_best_links(bag, data_dict["costs"], trees)
            if self._kde:
                for link in links[::-1]:
                    p = max(self.calc_link_probability(link), 0.001)
                    link.affinity *= p
                    if link.affinity < self.min_affinity:
                        links.remove(link)

            if self.window_size >= 1 and links:
                # Store selected edges for subsequent frames
                vecs = np.vstack([link.to_vector() for link in links])
                self._trees[ind_frame] = cKDTree(vecs)

            assemblies, assembled_ = self.build_assemblies(links)
            assembled.update(assembled_)

        # Remove invalid assemblies
        discarded = set(
            joint
            for joint in joints
            if joint.idx not in assembled and np.isfinite(joint.confidence)
        )
        for assembly in assemblies[::-1]:
            if 0 < assembly.n_links < self.min_n_links or not len(assembly):
                for link in assembly._links:
                    discarded.update((link.j1, link.j2))
                assemblies.remove(assembly)
        if 0 < self.max_overlap < 1:  # Non-maximum pose suppression
            if self._kde is not None:
                scores = [
                    -self.calc_assembly_mahalanobis_dist(ass) for ass in assemblies
                ]
            else:
                scores = [ass._affinity for ass in assemblies]
            lst = list(zip(scores, assemblies))
            assemblies = []
            while lst:
                temp = max(lst, key=lambda x: x[0])
                lst.remove(temp)
                assemblies.append(temp[1])
                for pair in lst[::-1]:
                    if temp[1].intersection_with(pair[1]) >= self.max_overlap:
                        lst.remove(pair)
        if len(assemblies) > self.max_n_individuals:
            assemblies = sorted(assemblies, key=len, reverse=True)
            for assembly in assemblies[self.max_n_individuals :]:
                for link in assembly._links:
                    discarded.update((link.j1, link.j2))
            assemblies = assemblies[: self.max_n_individuals]

        if self.add_discarded and discarded:
            # Fill assemblies with unconnected body parts
            for joint in sorted(discarded, key=lambda x: x.confidence, reverse=True):
                if self.safe_edge:
                    for assembly in assemblies:
                        if joint.label in assembly._visible:
                            continue
                        d_old = self.calc_assembly_mahalanobis_dist(assembly)
                        assembly.add_joint(joint)
                        d = self.calc_assembly_mahalanobis_dist(assembly)
                        if d < d_old:
                            break
                        assembly.remove_joint(joint)
                else:
                    dists = []
                    for i, assembly in enumerate(assemblies):
                        if joint.label in assembly._visible:
                            continue
                        d = cdist(assembly.xy, np.atleast_2d(joint.pos))
                        dists.append((i, np.nanmin(d)))
                    if not dists:
                        continue
                    min_ = sorted(dists, key=lambda x: x[1])
                    ind, _ = min_[0]
                    assemblies[ind].add_joint(joint)

        return assemblies, unique

    def assemble(self, chunk_size=1, n_processes=None):
        self.assemblies = dict()
        self.unique = dict()
        # Spawning (rather than forking) multiple processes does not
        # work nicely with the GUI or interactive sessions.
        # In that case, we fall back to the serial assembly.
        if chunk_size == 0 or multiprocessing.get_start_method() == "spawn":
            for i, data_dict in enumerate(tqdm(self)):
                assemblies, unique = self._assemble(data_dict, i)
                if assemblies:
                    self.assemblies[i] = assemblies
                if unique is not None:
                    self.unique[i] = unique
        else:
            global wrapped  # Hack to make the function pickable

            def wrapped(i):
                return i, self._assemble(self[i], i)

            n_frames = len(self.metadata["imnames"])
            with multiprocessing.Pool(n_processes) as p:
                with tqdm(total=n_frames) as pbar:
                    for i, (assemblies, unique) in p.imap_unordered(
                        wrapped, range(n_frames), chunksize=chunk_size
                    ):
                        if assemblies:
                            self.assemblies[i] = assemblies
                        if unique is not None:
                            self.unique[i] = unique
                        pbar.update()

    def from_pickle(self, pickle_path):
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)
        self.unique = data.pop("single", {})
        self.assemblies = data

    @staticmethod
    def parse_metadata(data):
        params = dict()
        params["joint_names"] = data["metadata"]["all_joints_names"]
        params["num_joints"] = len(params["joint_names"])
        params["paf_graph"] = data["metadata"]["PAFgraph"]
        params["paf"] = data["metadata"].get(
            "PAFinds", np.arange(len(params["joint_names"]))
        )
        params["bpts"] = params["ibpts"] = range(params["num_joints"])
        params["imnames"] = [fn for fn in list(data) if fn != "metadata"]
        return params

    def to_h5(self, output_name):
        data = np.full(
            (
                len(self.metadata["imnames"]),
                self.max_n_individuals,
                self.n_multibodyparts,
                4,
            ),
            fill_value=np.nan,
        )
        for ind, assemblies in self.assemblies.items():
            for n, assembly in enumerate(assemblies):
                data[ind, n] = assembly.data
        index = pd.MultiIndex.from_product(
            [
                ["scorer"],
                map(str, range(self.max_n_individuals)),
                map(str, range(self.n_multibodyparts)),
                ["x", "y", "likelihood"],
            ],
            names=["scorer", "individuals", "bodyparts", "coords"],
        )
        temp = data[..., :3].reshape((data.shape[0], -1))
        df = pd.DataFrame(temp, columns=index)
        df.to_hdf(output_name, key="ass")

    def to_pickle(self, output_name):
        data = dict()
        for ind, assemblies in self.assemblies.items():
            data[ind] = [ass.data for ass in assemblies]
        if self.unique:
            data["single"] = self.unique
        with open(output_name, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

class Joint:
    pos: Position
    confidence: float = 1.0
    label: int = None
    idx: int = None
    group: int = -1

class Link:
    def __init__(self, j1, j2, affinity=1):
        self.j1 = j1
        self.j2 = j2
        self.affinity = affinity
        self._length = sqrt((j1.pos[0] - j2.pos[0]) ** 2 + (j1.pos[1] - j2.pos[1]) ** 2)

    def __repr__(self):
        return (
            f"Link {self.idx}, affinity={self.affinity:.2f}, length={self.length:.2f}"
        )

    @property
    def confidence(self):
        return self.j1.confidence * self.j2.confidence

    @property
    def idx(self):
        return self.j1.idx, self.j2.idx

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, length):
        self._length = length

    def to_vector(self):
        return [*self.j1.pos, *self.j2.pos]

