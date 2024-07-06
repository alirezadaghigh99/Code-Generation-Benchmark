def from_bounds(cls, bounds: BoundsType, normalized: bool = False):
        """Creates and instance of this class from integration bounds.

        .. seealso::
            * :class:`emukit.quadrature.measures.BoxDomain`

        :param bounds: List of :math:`d` tuples :math:`[(a_1, b_1), (a_2, b_2), \dots, (a_d, b_d)]`,
                       where :math:`d` is the dimensionality of the domain and the tuple :math:`(a_i, b_i)`
                       contains the lower and upper bound of dimension :math:`i` defining the box domain.
        :param normalized: Weather the Lebesgue measure is normalized.
        :return: An instance of LebesgueMeasure.

        """
        domain = BoxDomain(name="", bounds=bounds)
        return cls(domain=domain, normalized=normalized)

