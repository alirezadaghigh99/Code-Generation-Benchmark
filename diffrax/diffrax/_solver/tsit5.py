class Tsit5(AbstractERK):
    r"""Tsitouras' 5/4 method.

    5th order explicit Runge--Kutta method. Has an embedded 4th order method for
    adaptive step sizing. Uses 7 stages with FSAL. Uses 5th order interpolation
    for dense/ts output.

    ??? cite "Reference"

        ```bibtex
        @article{tsitouras2011runge,
          title={Runge--Kutta pairs of order 5 (4) satisfying only the first column
                 simplifying assumption},
          author={Tsitouras, Ch},
          journal={Computers \& Mathematics with Applications},
          volume={62},
          number={2},
          pages={770--775},
          year={2011},
          publisher={Elsevier}
        }
        ```
    """

    tableau: ClassVar[ButcherTableau] = _tsit5_tableau
    interpolation_cls: ClassVar[
        Callable[..., _Tsit5Interpolation]
    ] = _Tsit5Interpolation

    def order(self, terms):
        return 5

