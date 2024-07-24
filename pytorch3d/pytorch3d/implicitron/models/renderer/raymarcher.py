class EmissionAbsorptionRaymarcher(AccumulativeRaymarcherBase):
    """
    Implements the EmissionAbsorption raymarcher.
    """

    background_opacity: float = 1e10

    @property
    def capping_function_type(self) -> str:
        return "exponential"

    @property
    def weight_function_type(self) -> str:
        return "product"

