import numpy as np

from cosipy.threeml import BinnedSED10


def _configured_spectrum(index=-2.0):
    spectrum = BinnedSED10()
    edges = np.geomspace(100.0, 10000.0, 11)

    for i, edge in enumerate(edges):
        getattr(spectrum, f"E{i}").value = float(edge)

    for i in range(10):
        getattr(spectrum, f"K{i}").value = 1e-6 * (i + 1)

    spectrum.index.value = index

    return spectrum, edges


def test_binned_sed10_evaluates_at_bin_pivots():
    spectrum, edges = _configured_spectrum()

    pivots = np.sqrt(edges[:-1] * edges[1:])
    flux = spectrum(pivots)
    expected = np.array([1e-6 * (i + 1) for i in range(10)])

    assert np.allclose(flux, expected)


def test_binned_sed10_zero_outside_range():
    spectrum, edges = _configured_spectrum()

    flux = spectrum(np.array([0.5 * edges[0], 2.0 * edges[-1]]))

    assert np.array_equal(flux, [0.0, 0.0])


def test_binned_sed10_integral_index_minus_one():
    spectrum, edges = _configured_spectrum(index=-1.0)

    expected = 0.0
    for i in range(10):
        k = getattr(spectrum, f"K{i}").value
        epiv = np.sqrt(edges[i] * edges[i + 1])
        expected += k * epiv * np.log(edges[i + 1] / edges[i])

    assert np.isclose(
        spectrum.integral(edges[0], edges[-1]),
        expected,
    )
