class ObjectPredictionList(Sequence):
    def __init__(self, list):
        self.list = list
        super().__init__()

    def __getitem__(self, i):
        if torch.is_tensor(i) or isinstance(i, np.ndarray):
            i = i.tolist()
        if isinstance(i, int):
            return ObjectPredictionList([self.list[i]])
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(self.list.__getitem__, i)
            return ObjectPredictionList(list(accessed_mapping))
        else:
            raise NotImplementedError(f"{type(i)}")

    def __setitem__(self, i, elem):
        if torch.is_tensor(i) or isinstance(i, np.ndarray):
            i = i.tolist()
        if isinstance(i, int):
            self.list[i] = elem
        elif isinstance(i, (tuple, list)):
            if len(i) != len(elem):
                raise ValueError()
            if isinstance(elem, ObjectPredictionList):
                for ind, el in enumerate(elem.list):
                    self.list[i[ind]] = el
            else:
                for ind, el in enumerate(elem):
                    self.list[i[ind]] = el
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self.list)

    def __str__(self):
        return str(self.list)

    def extend(self, object_prediction_list):
        self.list.extend(object_prediction_list.list)

    def totensor(self):
        return object_prediction_list_to_torch(self)

    def tonumpy(self):
        return object_prediction_list_to_numpy(self)

    def tolist(self):
        if len(self.list) == 1:
            return self.list[0]
        else:
            return self.list

