class _Dopri5Interpolation(FourthOrderPolynomialInterpolation):
    c_mid: ClassVar[np.ndarray] = np.array(
        [
            6025192743 / 30085553152 / 2,
            0,
            51252292925 / 65400821598 / 2,
            -2691868925 / 45128329728 / 2,
            187940372067 / 1594534317056 / 2,
            -1776094331 / 19743644256 / 2,
            11237099 / 235043384 / 2,
        ]
    )

