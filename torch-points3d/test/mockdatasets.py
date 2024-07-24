class MockDatasetConfig(object):
    def __init__(self):
        pass

    def keys(self):
        return []

    def get(self, dataset_name, default):
        return None

class MockDataset(torch.utils.data.Dataset):
    def __init__(self, feature_size=0, transform=None, num_points=100, panoptic=False, include_box=False, batch_size=2):
        self.feature_dimension = feature_size
        self.num_classes = 10
        self.num_points = num_points
        self.batch_size = batch_size
        self.weight_classes = None
        self.feature_size = feature_size
        if feature_size > 0:
            self._feature = torch.tensor([range(feature_size) for i in range(self.num_points)], dtype=torch.float,)
        else:
            self._feature = None
        self._y = torch.randint(0, 10, (self.num_points,))
        self._category = torch.ones((self.num_points,), dtype=torch.long)
        self._ms_transform = None
        self._transform = transform
        self.mean_size_arr = torch.rand((11, 3))
        self.include_box = include_box
        self.panoptic = panoptic

    def __len__(self):
        return self.num_points

    def len(self):
        return len(self)

    def _generate_data(self):
        data = Data(
            pos=torch.randn((self.num_points, 3)),
            x=torch.randn((self.num_points, self.feature_size)) if self.feature_size else None,
            y=torch.randint(0, 10, (self.num_points,)),
            category=self._category,
        )
        if self.include_box:
            num_boxes = 10
            data.center_label = torch.randn(num_boxes, 3)
            data.heading_class_label = torch.zeros((num_boxes,))
            data.heading_residual_label = torch.randn((num_boxes,))
            data.size_class_label = torch.randint(0, len(self.mean_size_arr), (num_boxes,))
            data.size_residual_label = torch.randn(num_boxes, 3)
            data.sem_cls_label = torch.randint(0, 10, (num_boxes,))
            data.box_label_mask = torch.randint(0, 1, (num_boxes,)).bool()
            data.vote_label = torch.randn(self.num_points, 9)
            data.vote_label_mask = torch.randint(0, 1, (self.num_points,)).bool()
            data.instance_box_corners = torch.randn((num_boxes, 8, 3)).bool()
        if self.panoptic:
            data.num_instances = torch.tensor([10])
            data.center_label = torch.randn((self.num_points, 3))
            data.y = torch.randint(0, 10, (self.num_points,))
            data.instance_labels = torch.randint(0, 20, (self.num_points,))
            data.instance_mask = torch.rand(self.num_points).bool()
            data.vote_label = torch.randn((self.num_points, 3))
        return data

    @property
    def datalist(self):
        datalist = [self._generate_data() for i in range(self.batch_size)]
        if self._transform:
            datalist = [self._transform(d.clone()) for d in datalist]
        if self._ms_transform:
            datalist = [self._ms_transform(d.clone()) for d in datalist]
        return datalist

    def __getitem__(self, index):
        return SimpleBatch.from_data_list(self.datalist)

    @property
    def class_to_segments(self):
        return {"class1": [0, 1, 2, 3, 4, 5], "class2": [6, 7, 8, 9]}

    @property
    def stuff_classes(self):
        return torch.tensor([0, 1])

    def set_strategies(self, model):
        strategies = model.get_spatial_ops()
        transform = MultiScaleTransform(strategies)
        self._ms_transform = transform

