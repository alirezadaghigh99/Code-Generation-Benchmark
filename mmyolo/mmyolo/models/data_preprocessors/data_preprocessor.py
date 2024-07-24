class PPYOLOEDetDataPreprocessor(DetDataPreprocessor):
    """Image pre-processor for detection tasks.

    The main difference between PPYOLOEDetDataPreprocessor and
    DetDataPreprocessor is the normalization order. The official
    PPYOLOE resize image first, and then normalize image.
    In DetDataPreprocessor, the order is reversed.

    Note: It must be used together with
    `mmyolo.datasets.utils.yolov5_collate`
    """

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization、padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``. This class use batch_augments first, and then
        normalize the image, which is different from the `DetDataPreprocessor`
        .

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        if not training:
            return super().forward(data, training)

        assert isinstance(data['inputs'], list) and is_list_of(
            data['inputs'], torch.Tensor), \
            '"inputs" should be a list of Tensor, but got ' \
            f'{type(data["inputs"])}. The possible reason for this ' \
            'is that you are not using it with ' \
            '"mmyolo.datasets.utils.yolov5_collate". Please refer to ' \
            '"cconfigs/ppyoloe/ppyoloe_plus_s_fast_8xb8-80e_coco.py".'

        data = self.cast_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        assert isinstance(data['data_samples'], dict)

        # Process data.
        batch_inputs = []
        for _input in inputs:
            # channel transform
            if self._channel_conversion:
                _input = _input[[2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _input = _input.float()
            batch_inputs.append(_input)

        # Batch random resize image.
        if self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(batch_inputs, data_samples)

        if self._enable_normalize:
            inputs = (inputs - self.mean) / self.std

        img_metas = [{'batch_input_shape': inputs.shape[2:]}] * len(inputs)
        data_samples = {
            'bboxes_labels': data_samples['bboxes_labels'],
            'img_metas': img_metas
        }

        return {'inputs': inputs, 'data_samples': data_samples}

class YOLOXBatchSyncRandomResize(BatchSyncRandomResize):
    """YOLOX batch random resize.

    Args:
        random_size_range (tuple): The multi-scale random range during
            multi-scale training.
        interval (int): The iter interval of change
            image size. Defaults to 10.
        size_divisor (int): Image size divisible factor.
            Defaults to 32.
    """

    def forward(self, inputs: Tensor, data_samples: dict) -> Tensor and dict:
        """resize a batch of images and bboxes to shape ``self._input_size``"""
        h, w = inputs.shape[-2:]
        inputs = inputs.float()
        assert isinstance(data_samples, dict)

        if self._input_size is None:
            self._input_size = (h, w)
        scale_y = self._input_size[0] / h
        scale_x = self._input_size[1] / w
        if scale_x != 1 or scale_y != 1:
            inputs = F.interpolate(
                inputs,
                size=self._input_size,
                mode='bilinear',
                align_corners=False)

            data_samples['bboxes_labels'][:, 2::2] *= scale_x
            data_samples['bboxes_labels'][:, 3::2] *= scale_y

            if 'keypoints' in data_samples:
                data_samples['keypoints'][..., 0] *= scale_x
                data_samples['keypoints'][..., 1] *= scale_y

        message_hub = MessageHub.get_current_instance()
        if (message_hub.get_info('iter') + 1) % self._interval == 0:
            self._input_size = self._get_random_size(
                aspect_ratio=float(w / h), device=inputs.device)

        return inputs, data_samples

