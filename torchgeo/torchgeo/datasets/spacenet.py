class SpaceNet7(SpaceNet):
    """SpaceNet 7: Multi-Temporal Urban Development Challenge.

    `SpaceNet 7 <https://spacenet.ai/sn7-challenge/>`_ is a dataset which
    consist of medium resolution (4.0m) satellite imagery mosaics acquired from
    Planet Labs’ Dove constellation between 2017 and 2020. It includes ≈ 24
    images (one per month) covering > 100 unique geographies, and comprises >
    40,000 km2 of imagery and exhaustive polygon labels of building footprints
    therein, totaling over 11M individual annotations.

    Dataset features:

    * No. of train samples: 1423
    * No. of test samples: 466
    * No. of building footprints: 11,080,000
    * Area Coverage: 41,000 sq km
    * Chip size: 1023 x 1023
    * GSD: ~4m

    Dataset format:

    * Imagery - Planet Dove GeoTIFF

        * mosaic.tif

    * Labels - GeoJSON

        * labels.geojson

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2102.04420

    .. versionadded:: 0.2
    """

    dataset_id = 'spacenet7'
    collection_md5_dict = {
        'sn7_train_source': '9f8cc109d744537d087bd6ff33132340',
        'sn7_train_labels': '16f873e3f0f914d95a916fb39b5111b5',
        'sn7_test_source': 'e97914f58e962bba3e898f08a14f83b2',
    }

    imagery = {'img': 'mosaic.tif'}
    chip_size = {'img': (1023, 1023)}

    label_glob = 'labels.geojson'

    def __init__(
        self,
        root: str = 'data',
        split: str = 'train',
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        download: bool = False,
        api_key: str | None = None,
        checksum: bool = False,
    ) -> None:
        """Initialize a new SpaceNet 7 Dataset instance.

        Args:
            root: root directory where dataset can be found
            split: split selection which must be in ["train", "test"]
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory.
            api_key: a RadiantEarth MLHub API key to use for downloading the dataset
            checksum: if True, check the MD5 of the downloaded files (may be slow)

        Raises:
            DatasetNotFoundError: If dataset is not found and *download* is False.
        """
        self.root = root
        self.split = split
        self.filename = self.imagery['img']
        self.transforms = transforms
        self.checksum = checksum

        assert split in {'train', 'test'}, 'Invalid split'

        if split == 'test':
            self.collections = ['sn7_test_source']
        else:
            self.collections = ['sn7_train_source', 'sn7_train_labels']

        to_be_downloaded = self._check_integrity()

        if to_be_downloaded:
            if not download:
                raise DatasetNotFoundError(self)
            else:
                self._download(to_be_downloaded, api_key)

        self.files = self._load_files(root)

    def _load_files(self, root: str) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Args:
            root: root dir of dataset

        Returns:
            list of dicts containing paths for images and labels (if train split)
        """
        files = []
        if self.split == 'train':
            imgs = sorted(
                glob.glob(os.path.join(root, 'sn7_train_source', '*', self.filename))
            )
            lbls = sorted(
                glob.glob(os.path.join(root, 'sn7_train_labels', '*', self.label_glob))
            )
            for img, lbl in zip(imgs, lbls):
                files.append({'image_path': img, 'label_path': lbl})
        else:
            imgs = sorted(
                glob.glob(os.path.join(root, 'sn7_test_source', '*', self.filename))
            )
            for img in imgs:
                files.append({'image_path': img})
        return files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data at that index
        """
        files = self.files[index]
        img, tfm, raster_crs = self._load_image(files['image_path'])
        h, w = img.shape[1:]

        ch, cw = self.chip_size['img']
        sample = {'image': img[:, :ch, :cw]}
        if self.split == 'train':
            mask = self._load_mask(files['label_path'], tfm, raster_crs, (h, w))
            sample['mask'] = mask[:ch, :cw]

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

