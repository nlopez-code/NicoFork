"""
Astromodels spectral function for a fixed 10-bin true-energy SED.

Each bin is represented by a local power law

    dN/dE = K_i * (E / E_piv,i)**index

with E_piv,i = sqrt(E_i * E_{i+1}).  The ten normalizations K0...K9
are independent fit parameters, while E0...E10 and the local index are
normally fixed from the COSI true-energy response.
"""

import numpy as np
import astropy.units as u

from astromodels.functions.function import Function1D, FunctionMeta


__all__ = ["BinnedSED10"]


class BinnedSED10(Function1D, metaclass=FunctionMeta):
    r"""
    description :
        Ten-bin piecewise power-law SED in true energy. Each bin has an
        independent differential normalization at its geometric-center pivot.
    latex : $K_i (E/E_{\mathrm{piv},i})^{\alpha}$
    parameters :
        K0 :
            desc : Differential normalization in true-energy bin 0
            initial value : 1e-6
            is_normalization : True
            min : 0
            max : 1e-2
            delta : 1e-7
        K1 :
            desc : Differential normalization in true-energy bin 1
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K2 :
            desc : Differential normalization in true-energy bin 2
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K3 :
            desc : Differential normalization in true-energy bin 3
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K4 :
            desc : Differential normalization in true-energy bin 4
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K5 :
            desc : Differential normalization in true-energy bin 5
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K6 :
            desc : Differential normalization in true-energy bin 6
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K7 :
            desc : Differential normalization in true-energy bin 7
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K8 :
            desc : Differential normalization in true-energy bin 8
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7
        K9 :
            desc : Differential normalization in true-energy bin 9
            initial value : 1e-6
            min : 0
            max : 1e-2
            delta : 1e-7

        E0 :
            desc : Lower edge of SED bin 0
            initial value : 1
            fix : yes
        E1 :
            desc : Edge between SED bins 0 and 1
            initial value : 2
            fix : yes
        E2 :
            desc : Edge between SED bins 1 and 2
            initial value : 3
            fix : yes
        E3 :
            desc : Edge between SED bins 2 and 3
            initial value : 4
            fix : yes
        E4 :
            desc : Edge between SED bins 3 and 4
            initial value : 5
            fix : yes
        E5 :
            desc : Edge between SED bins 4 and 5
            initial value : 6
            fix : yes
        E6 :
            desc : Edge between SED bins 5 and 6
            initial value : 7
            fix : yes
        E7 :
            desc : Edge between SED bins 6 and 7
            initial value : 8
            fix : yes
        E8 :
            desc : Edge between SED bins 7 and 8
            initial value : 9
            fix : yes
        E9 :
            desc : Edge between SED bins 8 and 9
            initial value : 10
            fix : yes
        E10 :
            desc : Upper edge of SED bin 9
            initial value : 11
            fix : yes

        index :
            desc : Local power-law index in every SED bin
            initial value : -2
            min : -10
            max : 10
            fix : yes
    """

    def _set_units(self, x_unit, y_unit):
        # Bin edges are energies.
        for i in range(11):
            getattr(self, f"E{i}").unit = x_unit

        # Each K_i is a differential flux normalization.
        for i in range(10):
            getattr(self, f"K{i}").unit = y_unit

        self.index.unit = u.dimensionless_unscaled

    @staticmethod
    def _value_in_unit(value, unit):
        """Return a plain numeric value in the requested unit when possible."""
        if isinstance(value, u.Quantity):
            return value.to_value(unit)
        return np.asarray(value, dtype=float)

    def evaluate(
        self,
        x,
        K0, K1, K2, K3, K4,
        K5, K6, K7, K8, K9,
        E0, E1, E2, E3, E4, E5,
        E6, E7, E8, E9, E10,
        index,
    ):
        # astromodels can evaluate with or without astropy quantities.
        x_has_units = isinstance(x, u.Quantity)

        if x_has_units:
            xv = np.asarray(x.to_value(self.x_unit), dtype=float)
            edges = np.array(
                [
                    self._value_in_unit(E0, self.x_unit),
                    self._value_in_unit(E1, self.x_unit),
                    self._value_in_unit(E2, self.x_unit),
                    self._value_in_unit(E3, self.x_unit),
                    self._value_in_unit(E4, self.x_unit),
                    self._value_in_unit(E5, self.x_unit),
                    self._value_in_unit(E6, self.x_unit),
                    self._value_in_unit(E7, self.x_unit),
                    self._value_in_unit(E8, self.x_unit),
                    self._value_in_unit(E9, self.x_unit),
                    self._value_in_unit(E10, self.x_unit),
                ],
                dtype=float,
            )
            kvals = np.array(
                [
                    self._value_in_unit(K0, self.y_unit),
                    self._value_in_unit(K1, self.y_unit),
                    self._value_in_unit(K2, self.y_unit),
                    self._value_in_unit(K3, self.y_unit),
                    self._value_in_unit(K4, self.y_unit),
                    self._value_in_unit(K5, self.y_unit),
                    self._value_in_unit(K6, self.y_unit),
                    self._value_in_unit(K7, self.y_unit),
                    self._value_in_unit(K8, self.y_unit),
                    self._value_in_unit(K9, self.y_unit),
                ],
                dtype=float,
            )
        else:
            xv = np.asarray(x, dtype=float)
            edges = np.asarray(
                [E0, E1, E2, E3, E4, E5, E6, E7, E8, E9, E10],
                dtype=float,
            )
            kvals = np.asarray(
                [K0, K1, K2, K3, K4, K5, K6, K7, K8, K9],
                dtype=float,
            )

        index_value = float(getattr(index, "value", index))

        if np.any(np.diff(edges) <= 0):
            raise ValueError("BinnedSED10 energy edges must be strictly increasing.")

        flux = np.zeros_like(xv, dtype=float)

        for i in range(10):
            elo = edges[i]
            ehi = edges[i + 1]
            epiv = np.sqrt(elo * ehi)

            # Half-open bins [E_i, E_{i+1}), except the last bin which
            # includes its upper edge. The exact endpoint convention has
            # zero effect on a continuous energy integral.
            if i < 9:
                mask = (xv >= elo) & (xv < ehi)
            else:
                mask = (xv >= elo) & (xv <= ehi)

            # Evaluate only inside the bin. This avoids unnecessary powers
            # outside the model support and keeps the piecewise definition clear.
            if np.any(mask):
                flux[mask] = kvals[i] * np.power(
                    xv[mask] / epiv,
                    index_value,
                )

        if x_has_units:
            return flux * self.y_unit

        return flux

    def integral(self, a, b):
        """
        Exact integral between two numerical energy boundaries.

        This follows the astromodels ``Function1D.integral`` convention and
        returns a plain numerical value.  Use ``Function1D.integrate`` when
        passing astropy quantities and a unit-bearing result is desired.
        """

        if isinstance(a, u.Quantity):
            a = a.to_value(self.x_unit)
        if isinstance(b, u.Quantity):
            b = b.to_value(self.x_unit)

        av = float(a)
        bv = float(b)

        if bv < av:
            return -self.integral(bv, av)

        edges = np.array(
            [getattr(self, f"E{i}").value for i in range(11)],
            dtype=float,
        )
        kvals = np.array(
            [getattr(self, f"K{i}").value for i in range(10)],
            dtype=float,
        )
        idx = float(self.index.value)

        if np.any(np.diff(edges) <= 0):
            raise ValueError("BinnedSED10 energy edges must be strictly increasing.")

        total = 0.0

        for i in range(10):
            lo = max(av, edges[i])
            hi = min(bv, edges[i + 1])

            if hi <= lo:
                continue

            epiv = np.sqrt(edges[i] * edges[i + 1])

            if np.isclose(idx, -1.0):
                integ = kvals[i] * epiv * np.log(hi / lo)
            else:
                integ = (
                    kvals[i]
                    * epiv
                    / (idx + 1.0)
                    * (
                        np.power(hi / epiv, idx + 1.0)
                        - np.power(lo / epiv, idx + 1.0)
                    )
                )

            total += integ

        return float(total)
