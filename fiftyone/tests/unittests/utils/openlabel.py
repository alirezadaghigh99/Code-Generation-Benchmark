def from_dict(cls, d):
        if d:
            object_type = list(d.keys())[0]
            obj_data = cls(None, None, object_type, as_list=False)
            obj_data._data_dict = d.pop(object_type, {})
            return obj_data
        else:
            return None

