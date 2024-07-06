def predict_instances(self, *args, **kwargs):
        # the reason why the actual computation happens as a generator function
        # (in '_predict_instances_generator') is that the generator is called
        # from the stardist napari plugin, which has its benefits regarding
        # control flow and progress display. however, typical use cases should
        # almost always use this function ('predict_instances'), and shouldn't
        # even notice (thanks to @functools.wraps) that it wraps the generator
        # function. note that similar reasoning applies to 'predict' and
        # 'predict_sparse'.

        # return last "yield"ed value of generator
        r = None
        for r in self._predict_instances_generator(*args, **kwargs):
            pass
        return r

