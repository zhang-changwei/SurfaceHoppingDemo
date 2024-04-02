import numpy as np
from numpy.typing import ArrayLike

from modelbase import DiabaticModel_

class TullySimpleAvoidedCrossing(DiabaticModel_):
    r"""Tunneling through a single barrier model used in Tully's 1990 JCP

    .. math::
       V_{11} &= \left\{ \begin{array}{cr}
                     A (1 - e^{Bx}) & x < 0 \\
                    -A (1 - e^{-Bx}) & x > 0
                     \end{array} \right. \\
       V_{22} &= -V_{11} \\
       V_{12} &= V_{21} = C e^{-D x^2}

    """
    def __init__(self,
                 representation: str = "adiabatic",
                 a: float = 0.01,
                 b: float = 1.6,
                 c: float = 0.005,
                 d: float = 1.0,
                 mass: float = 2000.0):
        """Constructor that defaults to the values reported in Tully's 1990 JCP"""
        DiabaticModel_.__init__(self, representation=representation,
                                nstate=2, ndim=1)

        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim())

    def V(self, X: ArrayLike) -> ArrayLike:
        """:math:`V(x)`"""
        x = X[0]
        v11 = float(np.copysign(self.A, x) * (1.0 - np.exp(-self.B * np.abs(x))))
        v22 = -v11
        v12 = float(self.C * np.exp(-self.D * x * x))
        out = np.array([[v11, v12], [v12, v22]], dtype=np.float64)
        return out

    def dV(self, x: ArrayLike) -> ArrayLike:
        """:math:`\\nabla V(x)`"""
        xx = np.array(x, dtype=np.float64)
        v11 = self.A * self.B * np.exp(-self.B * abs(xx))
        v22 = -v11
        v12 = -2.0 * self.C * self.D * xx * np.exp(-self.D * xx * xx)
        out = np.array([[v11, v12], [v12, v22]], dtype=np.float64)
        return out.reshape([1, 2, 2])


class TullyDualAvoidedCrossing(DiabaticModel_):
    r"""Tunneling through a double avoided crossing used in Tully's 1990 JCP

    .. math::
        V_{11} &= 0 \\
        V_{22} &= -A e^{-Bx^2} + E_0 \\
        V_{12} &= V_{21} = C e^{-D x^2}
    """
    def __init__(self,
                 representation: str = "adiabatic",
                 a: float = 0.1,
                 b: float = 0.28,
                 c: float = 0.015,
                 d: float = 0.06,
                 e: float = 0.05,
                 mass: float = 2000.0):
        """Constructor that defaults to the values reported in Tully's 1990 JCP"""
        DiabaticModel_.__init__(self, representation=representation,
                                nstate=2, ndim=1)
        self.A = a
        self.B = b
        self.C = c
        self.D = d
        self.E0 = e
        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim())

    def V(self, X: ArrayLike) -> ArrayLike:
        """:math:`V(x)`"""
        x = X[0]
        v11 = 0.0
        v22 = float(-self.A * np.exp(-self.B * x * x) + self.E0)
        v12 = float(self.C * np.exp(-self.D * x * x))
        out = np.array([[v11, v12], [v12, v22]], dtype=np.float64)
        return out

    def dV(self, x: ArrayLike) -> ArrayLike:
        """:math:`\\nabla V(x)`"""
        xx = np.array(x, dtype=np.float64)
        v11 = np.zeros_like(xx)
        v22 = 2.0 * self.A * self.B * xx * np.exp(-self.B * xx * xx)
        v12 = -2.0 * self.C * self.D * xx * np.exp(-self.D * xx * xx)
        out = np.array([[v11, v12], [v12, v22]], dtype=np.float64)
        return out.reshape([1, 2, 2])


class TullyExtendedCouplingReflection(DiabaticModel_):
    r"""Model with extended coupling and the possibility of reflection. The most challenging of the
    models used in Tully's 1990 JCP

    .. math::
        V_{11} &= A \\
        V_{22} &= -A \\
        V_{12} &= \left\{ \begin{array}{cr}
                      B e^{Cx} & x < 0 \\
                      B \left( 2 - e^{-Cx} \right) & x > 0
                      \end{array} \right.
    """

    def __init__(self,
                 representation: str = "adiabatic",
                 a: float = 0.0006,
                 b: float = 0.10,
                 c: float = 0.90,
                 mass: float = 2000.0):
        """Constructor that defaults to the values reported in Tully's 1990 JCP"""
        DiabaticModel_.__init__(self, representation=representation,
                                ndim=1, nstate=2)
        self.A = a
        self.B = b
        self.C = c
        self.mass = np.array(mass, dtype=np.float64).reshape(self.ndim())

    def V(self, X: ArrayLike) -> ArrayLike:
        """:math:`V(x)`"""
        x = X[0]
        v11 = self.A
        v22 = -self.A
        v12 = float(np.exp(-np.abs(x) * self.C))
        if x < 0:
            v12 = self.B * v12
        else:
            v12 = self.B * (2.0 - v12)
        out = np.array([[v11, v12], [v12, v22]], dtype=np.float64)
        return out

    def dV(self, x: ArrayLike) -> ArrayLike:
        """:math:`\\nabla V(x)`"""
        xx = np.array(x, dtype=np.float64)
        v11 = np.zeros_like(xx)
        v22 = np.zeros_like(xx)
        v12 = self.B * self.C * np.exp(-self.C * np.abs(xx))
        out = np.array([[v11, v12], [v12, v22]], dtype=np.float64)
        return out.reshape([1, 2, 2])
