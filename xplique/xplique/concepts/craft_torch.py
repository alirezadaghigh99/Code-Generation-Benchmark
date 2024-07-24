class CraftTorch(BaseCraft):
    """
    Class Implementing the CRAFT Concept Extraction Mechanism on Pytorch.

    Parameters
    ----------
    input_to_latent_model
        The first part of the model taking an input and returning
        positive activations, g(.) in the original paper.
        Must be a Pytorch model (torch.nn.modules.module.Module) accepting
        data of shape (n_samples, channels, height, width).
    latent_to_logit_model
        The second part of the model taking activation and returning
        logits, h(.) in the original paper.
        Must be a Pytorch model (torch.nn.modules.module.Module).
    number_of_concepts
        The number of concepts to extract. Default is 20.
    batch_size
        The batch size to use during training and prediction. Default is 64.
    patch_size
        The size of the patches (crops) to extract from the input data. Default is 64.
    device
        The device to use. Default is 'cuda'.
    """

    def __init__(self, input_to_latent_model: Callable,
                       latent_to_logit_model: Callable,
                       number_of_concepts: int = 20,
                       batch_size: int = 64,
                       patch_size: int = 64,
                       device : str = 'cuda'):
        super().__init__(input_to_latent_model, latent_to_logit_model,
                         number_of_concepts, batch_size)
        self.patch_size = patch_size
        self.device = device

        # Check model type
        is_torch_model = issubclass(type(input_to_latent_model), torch.nn.modules.module.Module) & \
                         issubclass(type(latent_to_logit_model), torch.nn.modules.module.Module)
        if not is_torch_model:
            raise TypeError('input_to_latent_model and latent_to_logit_model are not ' \
                            'Pytorch modules')

    def _latent_predict(self, inputs: torch.Tensor, resize=None) -> torch.Tensor:
        """
        Compute the embedding space using the 1st model `input_to_latent_model`.

        Parameters
        ----------
        inputs
            Input data of shape (n_samples, channels, height, width).

        Returns
        -------
        activations
            The latent activations of shape (n_samples, height, width, channels)
        """
        # inputs: (N, C, H, W)
        if len(inputs.shape) == 3:
            inputs = inputs.unsqueeze(0) # add an extra dim in case we get only 1 image to predict

        activations = _batch_inference(self.input_to_latent_model, inputs,
                                       self.batch_size, resize, device=self.device)
        if len(activations.shape) == 4:
            # activations: (N, C, H, W) -> (N, H, W, C)
            activations = activations.permute(0, 2, 3, 1)
        return activations

    def _logit_predict(self, activations: np.ndarray, resize=None) -> torch.Tensor:
        """
        Compute logits from activations using the 2nd model `latent_to_logit_model`.

        Parameters
        ----------
        activations
            Activations produced by the 1st model `input_to_latent_model`,
            of shape (n_samples, height, width, channels).

        Returns
        -------
        logits
            The logits of shape (n_samples, n_classes)
        """
        # pylint disable=no-member
        activations_perturbated = torch.from_numpy(activations)

        if len(activations_perturbated.shape) == 4:
            # activations_perturbated: (N, H, W, C) -> (N, C, H, W)
            activations_perturbated = activations_perturbated.permute(0, 3, 1, 2)

        y_pred = _batch_inference(self.latent_to_logit_model, activations_perturbated,
                                  self.batch_size, resize, device=self.device)
        return self._to_np_array(y_pred)

    def _extract_patches(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Extract patches (crops) from the input images, and compute their embeddings.

        Parameters
        ----------
        inputs
            Input images (n_samples, channels, height, width)

        Returns
        -------
        patches
            A tuple containing the patches (n_patches, channels, height, width).
        activations
            The patches activations (n_patches, channels).
        """

        image_size = (inputs.shape[2], inputs.shape[3])
        num_channels = inputs.shape[1]

        # extract patches from the input data, keep patches on cpu
        strides = int(self.patch_size * 0.80)

        patches = torch.nn.functional.unfold(inputs, kernel_size=self.patch_size, stride=strides)
        patches = patches.transpose(1, 2).contiguous().view(-1, num_channels,
                                                            self.patch_size, self.patch_size)

        # encode the patches and obtain the activations
        activations = self._latent_predict(patches, resize=image_size)

        # pylint disable=no-member
        assert torch.min(activations) >= 0.0, "Activations must be positive."

        # if the activations have shape (n_samples, height, width, n_channels),
        # apply average pooling
        if len(activations.shape) == 4:
            # activations: (N, H, W, R)
            # pylint disable=no-member
            activations = torch.mean(activations, dim=(1, 2))

        return self._to_np_array(patches), self._to_np_array(activations)

    def _to_np_array(self, inputs: torch.Tensor, dtype: type = None):
        """
        Converts a Pytorch tensor into a numpy array.
        """
        res = inputs.detach().cpu().numpy()
        if dtype is not None:
            return res.astype(dtype)
        return res

