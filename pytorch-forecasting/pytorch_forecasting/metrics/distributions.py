class NegativeBinomialDistributionLoss(DistributionLoss):
    """
    Negative binomial loss, e.g. for count data.

    Requirements for original target normalizer:
        * not centered normalization (only rescaled)
    """

    distribution_class = distributions.NegativeBinomial
    distribution_arguments = ["mean", "shape"]

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.NegativeBinomial:
        mean = x[..., 0]
        shape = x[..., 1]
        r = 1.0 / shape
        p = mean / (mean + r)
        return self.distribution_class(total_count=r, probs=p)

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        assert not encoder.center, "NegativeBinomialDistributionLoss is not compatible with `center=True` normalization"
        assert encoder.transformation not in ["logit", "log"], "Cannot use bound transformation such as 'logit'"
        if encoder.transformation in ["log1p"]:
            mean = torch.exp(parameters[..., 0] * target_scale[..., 1].unsqueeze(-1))
            shape = (
                F.softplus(torch.exp(parameters[..., 1]))
                / torch.exp(target_scale[..., 1].unsqueeze(-1)).sqrt()  # todo: is this correct?
            )
        else:
            mean = F.softplus(parameters[..., 0]) * target_scale[..., 1].unsqueeze(-1)
            shape = F.softplus(parameters[..., 1]) / target_scale[..., 1].unsqueeze(-1).sqrt()
        return torch.stack([mean, shape], dim=-1)

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction. In the case of this distribution prediction we
        need to derive the mean (as a point prediction) from the distribution parameters

        Args:
            y_pred: prediction output of network
            in this case the two parameters for the negative binomial

        Returns:
            torch.Tensor: mean prediction
        """
        return y_pred[..., 0]

class MQF2DistributionLoss(DistributionLoss):
    """Multivariate quantile loss based on the article
    `Multivariate Quantile Function Forecaster <http://arxiv.org/abs/2202.11316>`_.

    Requires install of additional library:
    ``pip install pytorch-forecasting[mqf2]``
    """

    eps = 1e-4

    def __init__(
        self,
        prediction_length: int,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        hidden_size: Optional[int] = 4,
        es_num_samples: int = 50,
        beta: float = 1.0,
        icnn_hidden_size: int = 20,
        icnn_num_layers: int = 2,
        estimate_logdet: bool = False,
    ) -> None:
        """
        Args:
            prediction_length (int): maximum prediction length.
            quantiles (List[float], optional): default quantiles to output.
                Defaults to [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98].
            hidden_size (int, optional): hidden size per prediction length. Defaults to 4.
            es_num_samples (int, optional): Number of samples to calculate energy score.
                If None, maximum liklihood is used as opposed to energy score for optimization.
                Defaults to 50.
            beta (float, optional): between 0 and 1.0 to control how scale sensitive metric is (1=fully sensitive).
                Defaults to 1.0.
            icnn_hidden_size (int, optional): hidden size of distribution estimating network. Defaults to 20.
            icnn_num_layers (int, optional): number of hidden layers in distribution estimating network. Defaults to 2.
            estimate_logdet (bool, optional): if to estimate log determinant. Defaults to False.
        """
        super().__init__(quantiles=quantiles)

        from cpflows.flows import ActNorm
        from cpflows.icnn import PICNN

        from pytorch_forecasting.metrics._mqf2_utils import (
            DeepConvexNet,
            MQF2Distribution,
            SequentialNet,
            TransformedMQF2Distribution,
        )

        self.distribution_class = MQF2Distribution
        self.transformed_distribution_class = TransformedMQF2Distribution
        self.distribution_arguments = list(range(int(hidden_size)))
        self.prediction_length = prediction_length
        self.es_num_samples = es_num_samples
        self.beta = beta

        # define picnn
        convexnet = PICNN(
            dim=prediction_length,
            dimh=icnn_hidden_size,
            dimc=hidden_size * prediction_length,
            num_hidden_layers=icnn_num_layers,
            symm_act_first=True,
        )
        deepconvexnet = DeepConvexNet(
            convexnet,
            prediction_length,
            is_energy_score=self.is_energy_score,
            estimate_logdet=estimate_logdet,
        )

        if self.is_energy_score:
            networks = [deepconvexnet]
        else:
            networks = [
                ActNorm(prediction_length),
                deepconvexnet,
                ActNorm(prediction_length),
            ]

        self.picnn = SequentialNet(networks)

    @property
    def is_energy_score(self) -> bool:
        return self.es_num_samples is not None

    def map_x_to_distribution(self, x: torch.Tensor) -> distributions.Distribution:
        distr = self.distribution_class(
            picnn=self.picnn,
            hidden_state=x[..., :-2],
            prediction_length=self.prediction_length,
            is_energy_score=self.is_energy_score,
            es_num_samples=self.es_num_samples,
            beta=self.beta,
        )
        # rescale
        loc = x[..., -2][:, None]
        scale = x[..., -1][:, None]
        scaler = distributions.AffineTransform(loc=loc, scale=scale)
        if self._transformation is None:
            return self.transformed_distribution_class(distr, [scaler])
        else:
            return self.transformed_distribution_class(
                distr,
                [scaler, TorchNormalizer.get_transform(self._transformation)["inverse_torch"]],
            )

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative likelihood

        Args:
            y_pred: network output
            y_actual: actual values

        Returns:
            torch.Tensor: metric value on which backpropagation can be applied
        """
        distribution = self.map_x_to_distribution(y_pred)
        if self.is_energy_score:
            loss = distribution.energy_score(y_actual)
        else:
            loss = -distribution.log_prob(y_actual)
        return loss.reshape(-1, 1)

    def rescale_parameters(
        self, parameters: torch.Tensor, target_scale: torch.Tensor, encoder: BaseEstimator
    ) -> torch.Tensor:
        self._transformation = encoder.transformation
        return torch.concat([parameters.reshape(parameters.size(0), -1), target_scale], dim=-1)

    def to_quantiles(self, y_pred: torch.Tensor, quantiles: List[float] = None) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network
            quantiles (List[float], optional): quantiles for probability range. Defaults to quantiles as
                as defined in the class initialization.

        Returns:
            torch.Tensor: prediction quantiles (last dimension)
        """
        if quantiles is None:
            quantiles = self.quantiles
        distribution = self.map_x_to_distribution(y_pred)
        alpha = (
            torch.as_tensor(quantiles, device=y_pred.device)[:, None]
            .repeat(y_pred.size(0), 1)
            .expand(-1, self.prediction_length)
        )
        hidden_state = distribution.base_dist.hidden_state.repeat_interleave(len(quantiles), dim=0)
        result = distribution.quantile(alpha, hidden_state=hidden_state)  # (batch_size * quantiles x prediction_length)

        # reshape
        result = result.reshape(-1, len(quantiles), self.prediction_length).transpose(
            1, 2
        )  # (batch_size, prediction_length, quantile_size)

        return result

