class MockRCNNInference(object):
    """Use to mock the GeneralizedRCNN.inference()"""

    def __init__(self, image_size, resize_size):
        self.image_size = image_size
        self.resize_size = resize_size

    @property
    def device(self):
        return torch.device("cpu")

    def __call__(
        self,
        batched_inputs,
        detected_instances=None,
        do_postprocess: bool = True,
    ):
        return self.inference(
            batched_inputs,
            detected_instances,
            do_postprocess,
        )

    def inference(
        self,
        batched_inputs,
        detected_instances=None,
        do_postprocess: bool = True,
    ):
        scale_xy = (
            _get_scale_xy(self.image_size, self.resize_size) if do_postprocess else None
        )
        results = get_detected_instances_from_image(batched_inputs, scale_xy=scale_xy)
        # when do_postprocess is True, the result instances is stored inside a dict
        if do_postprocess:
            results = [{"instances": r} for r in results]

        return results

