from types import SimpleNamespace

import numpy as np
import pytest
import astropy.units as u

from cosipy.threeml import (
    BinnedSED10,
    check_binned_sed10_response,
    configure_binned_sed10_from_response,
    find_unconstrained_sed_bins,
    freeze_binned_sed10_bins,
    profile_likelihood_upper_limit,
)


class _DummyEiAxis:
    def __init__(self):
        self.edges = np.geomspace(100.0, 10000.0, 16) * u.keV
        self.nbins = len(self.edges) - 1


class _DummyResponse:
    def __init__(self):
        self.axes = {"Ei": _DummyEiAxis()}


class _DummyExpectation:
    def __init__(self, contents):
        self.contents = np.asarray(contents, dtype=float)


class _DummySourceResponse:
    def __init__(self, spectrum, zero_bin=3):
        self._source = SimpleNamespace(
            spectrum=SimpleNamespace(
                main=SimpleNamespace(shape=spectrum)
            )
        )
        self.zero_bin = zero_bin

    def expectation(self, copy=False):
        values = np.array([
            getattr(self._source.spectrum.main.shape, f"K{i}").value
            for i in range(10)
        ])

        active = int(np.argmax(values))

        if values[active] == 0.0 or active == self.zero_bin:
            return _DummyExpectation(np.zeros((2, 2)))

        return _DummyExpectation(np.full((2, 2), values[active]))


def test_configure_binned_sed10_from_response():
    spectrum = BinnedSED10()
    response = _DummyResponse()
    initial_fluxes = np.geomspace(1e-8, 1e-6, 10)

    configure_binned_sed10_from_response(
        spectrum,
        response,
        ei_bin_indices=range(2, 12),
        initial_fluxes=initial_fluxes,
        index=-2.0,
    )

    expected_edges = response.axes["Ei"].edges[2:13].to_value(u.keV)
    actual_edges = np.array([
        getattr(spectrum, f"E{i}").value
        for i in range(11)
    ])
    actual_fluxes = np.array([
        getattr(spectrum, f"K{i}").value
        for i in range(10)
    ])

    assert np.allclose(actual_edges, expected_edges)
    assert np.allclose(actual_fluxes, initial_fluxes)
    assert spectrum.index.value == -2.0
    assert spectrum._cosipy_ei_bin_indices == tuple(range(2, 12))
    assert all(getattr(spectrum, f"K{i}").free for i in range(10))


def test_find_and_freeze_unconstrained_sed_bins():
    templates = np.ones((4, 3, 2))
    templates[1] = 0.0
    templates[3] = 0.0

    bins = find_unconstrained_sed_bins(
        templates,
        bin_indices=[5, 6, 7, 8],
    )

    assert np.array_equal(bins, [6, 8])

    spectrum = BinnedSED10()
    frozen = freeze_binned_sed10_bins(spectrum, [2, 7])

    assert frozen == (2, 7)
    assert spectrum.K2.value == 0.0
    assert spectrum.K7.value == 0.0
    assert spectrum.K2.free is False
    assert spectrum.K7.free is False


def test_check_binned_sed10_response():
    spectrum = BinnedSED10()
    spectrum._cosipy_ei_bin_indices = tuple(range(10, 20))
    source_response = _DummySourceResponse(spectrum, zero_bin=3)

    original_values = np.array([
        getattr(spectrum, f"K{i}").value
        for i in range(10)
    ])

    with pytest.warns(UserWarning):
        result = check_binned_sed10_response(
            spectrum,
            source_response,
            freeze=True,
            verbose=False,
        )

    assert np.array_equal(result["local_indices"], [3])
    assert np.array_equal(result["response_indices"], [13])
    assert result["template_totals"][3] == 0.0

    for i in range(10):
        par = getattr(spectrum, f"K{i}")
        if i == 3:
            assert par.value == 0.0
            assert par.free is False
        else:
            assert par.value == original_values[i]


def test_profile_likelihood_upper_limit():
    best = 2.0e-6
    sigma = 0.4e-6
    nll_best = 12.3

    def profile_nll(value):
        return nll_best + 0.5 * ((value - best) / sigma) ** 2

    ul = profile_likelihood_upper_limit(
        profile_nll,
        nll_best=nll_best,
        best_value=best,
        sigma=sigma,
    )

    expected = best + sigma * np.sqrt(2.705543)

    assert np.isclose(ul, expected, rtol=1e-5)
