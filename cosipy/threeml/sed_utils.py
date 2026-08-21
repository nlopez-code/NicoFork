"""
Utilities for the fixed 10-bin COSI true-energy SED model.

The spectral model itself lives in :mod:`cosipy.threeml.binned_sed`.
This module contains response/fit helpers that are deliberately kept outside
of the astromodels function because they depend on the instrument response or
likelihood rather than on the mathematical spectrum alone.
"""

import warnings

import numpy as np
import astropy.units as u
from scipy.optimize import brentq

from .binned_sed import BinnedSED10


__all__ = [
    "configure_binned_sed10_from_response",
    "find_unconstrained_sed_bins",
    "freeze_binned_sed10_bins",
    "check_binned_sed10_response",
    "profile_likelihood_upper_limit",
]


def _dense_contents(hist):
    """Return histogram contents as a dense floating-point ndarray."""
    contents = hist.contents
    if hasattr(contents, "todense"):
        contents = contents.todense()
    return np.asarray(contents, dtype=float)


def _quantity_to_value(value, unit=None):
    """Convert a scalar/array Quantity to plain values when necessary."""
    if isinstance(value, u.Quantity):
        if unit is None:
            return np.asarray(value.value, dtype=float)
        return np.asarray(value.to_value(unit), dtype=float)
    return np.asarray(value, dtype=float)


def configure_binned_sed10_from_response(
    spectrum,
    response,
    ei_bin_indices,
    initial_fluxes=None,
    initial_spectrum=None,
    index=-2.0,
    default_initial_flux=1e-8,
):
    """
    Configure ``BinnedSED10`` from ten contiguous COSI response Ei bins.

    Parameters
    ----------
    spectrum : BinnedSED10
        Spectrum instance to configure.
    response : ExtendedSourceResponse or compatible object
        Must expose ``response.axes["Ei"].edges``.
    ei_bin_indices : iterable of int
        Exactly ten contiguous true-energy bin indices.
    initial_fluxes : array-like or Quantity, optional
        Initial values for K0...K9.  Cannot be supplied together with
        ``initial_spectrum``.
    initial_spectrum : astromodels Function1D, optional
        Broadband spectrum used to initialize K0...K9 at the geometric-center
        pivot of each selected Ei bin.
    index : float, optional
        Fixed local power-law index used in all ten bins. Default is -2.
    default_initial_flux : float, optional
        Positive fallback used when an initial flux is non-finite or non-positive.

    Returns
    -------
    BinnedSED10
        The configured input spectrum.
    """

    if not isinstance(spectrum, BinnedSED10):
        raise TypeError("spectrum must be a BinnedSED10 instance.")

    bins = np.asarray(list(ei_bin_indices), dtype=int)

    if bins.size != 10:
        raise ValueError("BinnedSED10 requires exactly 10 Ei bins.")

    if not np.all(np.diff(bins) == 1):
        raise ValueError("The 10 Ei bins must be contiguous and increasing.")

    ei_axis = response.axes["Ei"]
    if bins[0] < 0 or bins[-1] >= ei_axis.nbins:
        raise IndexError(
            f"Selected Ei bins must lie in [0, {ei_axis.nbins - 1}]."
        )

    edges = ei_axis.edges

    # COSI response energies are conventionally keV. If astromodels has already
    # assigned an x unit to the spectrum, use that instead.
    target_energy_unit = spectrum.x_unit if spectrum.x_unit is not None else u.keV

    if isinstance(edges, u.Quantity):
        selected_edges = np.asarray(
            edges[bins[0]: bins[-1] + 2].to_value(target_energy_unit),
            dtype=float,
        )
        selected_edges_quantity = selected_edges * target_energy_unit
    else:
        selected_edges = np.asarray(
            edges[bins[0]: bins[-1] + 2],
            dtype=float,
        )
        selected_edges_quantity = selected_edges * u.keV

    if np.any(np.diff(selected_edges) <= 0.0):
        raise ValueError("Selected response Ei edges are not strictly increasing.")

    for i, edge in enumerate(selected_edges):
        par = getattr(spectrum, f"E{i}")
        par.value = float(edge)
        par.free = False

    spectrum.index.value = float(index)
    spectrum.index.free = False

    if initial_fluxes is not None and initial_spectrum is not None:
        raise ValueError(
            "Specify either initial_fluxes or initial_spectrum, not both."
        )

    if initial_spectrum is not None:
        pivots = np.sqrt(
            selected_edges_quantity[:-1] * selected_edges_quantity[1:]
        )
        values = initial_spectrum(pivots)

        # If BinnedSED10 already has assigned output units, convert explicitly.
        if isinstance(values, u.Quantity) and spectrum.y_unit is not None:
            initial_fluxes = values.to_value(spectrum.y_unit)
        else:
            initial_fluxes = getattr(values, "value", values)

    if initial_fluxes is not None:
        if isinstance(initial_fluxes, u.Quantity):
            if spectrum.y_unit is not None:
                initial_fluxes = initial_fluxes.to_value(spectrum.y_unit)
            else:
                initial_fluxes = initial_fluxes.value

        initial_fluxes = np.asarray(initial_fluxes, dtype=float)

        if initial_fluxes.size != 10:
            raise ValueError("initial_fluxes must contain exactly 10 values.")

        if not np.isfinite(default_initial_flux) or default_initial_flux <= 0.0:
            raise ValueError("default_initial_flux must be finite and positive.")

        initial_fluxes = np.where(
            np.isfinite(initial_fluxes) & (initial_fluxes > 0.0),
            initial_fluxes,
            float(default_initial_flux),
        )

        for i, flux in enumerate(initial_fluxes):
            getattr(spectrum, f"K{i}").value = float(flux)

    for i in range(10):
        getattr(spectrum, f"K{i}").free = True

    # Retain the mapping as convenience metadata for the response diagnostic.
    # It is not a fit parameter and is not required for model evaluation.
    spectrum._cosipy_ei_bin_indices = tuple(int(i) for i in bins)

    return spectrum


def find_unconstrained_sed_bins(templates, bin_indices=None, atol=0.0):
    """
    Identify SED bins whose forward-folded source template is zero.

    Parameters
    ----------
    templates : array-like
        Array with shape ``(n_sed_bins, ...)``. The remaining dimensions are
        arbitrary detector-space axes.
    bin_indices : iterable of int, optional
        Labels to return for each template. Defaults to ``0..n_sed_bins-1``.
    atol : float, optional
        A bin is considered unconstrained when the sum of the absolute template
        values is less than or equal to ``atol``. Default is exactly zero.

    Returns
    -------
    numpy.ndarray
        Labels of unconstrained bins.
    """

    templates = np.asarray(templates, dtype=float)

    if templates.ndim < 2:
        raise ValueError(
            "templates must have shape (n_sed_bins, ...detector axes...)."
        )

    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    if not np.all(np.isfinite(templates)):
        bad = np.where(
            ~np.all(np.isfinite(templates.reshape(templates.shape[0], -1)), axis=1)
        )[0]
        raise ValueError(
            f"Non-finite values were found in SED templates {bad.tolist()}."
        )

    n_bins = templates.shape[0]

    if bin_indices is None:
        labels = np.arange(n_bins, dtype=int)
    else:
        labels = np.asarray(list(bin_indices), dtype=int)
        if labels.size != n_bins:
            raise ValueError("bin_indices must contain one label per template.")

    totals = np.sum(
        np.abs(templates.reshape(n_bins, -1)),
        axis=1,
    )

    return labels[totals <= atol]


def freeze_binned_sed10_bins(spectrum, local_bin_indices):
    """
    Freeze selected local BinnedSED10 normalizations to zero.

    Parameters
    ----------
    spectrum : BinnedSED10
        SED spectrum.
    local_bin_indices : iterable of int
        Local SED-bin indices in the range 0--9.

    Returns
    -------
    tuple of int
        Local bin indices that were frozen.
    """

    if not isinstance(spectrum, BinnedSED10):
        raise TypeError("spectrum must be a BinnedSED10 instance.")

    bins = tuple(sorted(set(int(i) for i in local_bin_indices)))

    for i in bins:
        if i < 0 or i >= 10:
            raise IndexError("BinnedSED10 local bin indices must lie in [0, 9].")

        par = getattr(spectrum, f"K{i}")
        par.value = 0.0
        par.free = False

    return bins


def _positive_test_value(parameter, default=1e-6):
    """Choose a positive in-bounds normalization for response diagnostics."""
    value = float(parameter.value)

    if np.isfinite(value) and value > 0.0:
        return value

    candidate = float(default)
    lo = parameter.min_value
    hi = parameter.max_value

    if hi is not None and candidate > hi:
        candidate = 0.5 * float(hi)

    if lo is not None and candidate <= lo:
        lo = float(lo)
        if hi is not None:
            candidate = lo + 0.5 * (float(hi) - lo)
        else:
            candidate = max(np.nextafter(lo, np.inf), default)

    if not np.isfinite(candidate) or candidate <= 0.0:
        raise ValueError(
            f"Could not choose a positive test value for parameter {parameter.path}."
        )

    return candidate


def check_binned_sed10_response(
    spectrum,
    source_response,
    atol=0.0,
    freeze=True,
    verbose=True,
):
    """
    Forward-fold each BinnedSED10 bin and identify zero-response bins.

    The function temporarily sets one K_i at a time to a positive value while
    setting the other nine normalizations to zero.  It then evaluates the
    supplied ``BinnedThreeMLExtendedSourceResponse``. With the spatial-response
    cache enabled, the expensive sky contraction is performed only once and the
    ten diagnostic folds are inexpensive.

    Parameters
    ----------
    spectrum : BinnedSED10
        Spectrum attached to the source used by ``source_response``.
    source_response : BinnedThreeMLExtendedSourceResponse
        Source response whose ``set_source`` method has already been called.
    atol : float, optional
        Template absolute-sum threshold defining an unconstrained bin.
    freeze : bool, optional
        If True, set each zero-response K_i to zero and freeze it.
    verbose : bool, optional
        Print a compact diagnostic for each SED bin.

    Returns
    -------
    dict
        Dictionary containing ``local_indices``, ``response_indices``, and
        ``template_totals``.
    """

    if not isinstance(spectrum, BinnedSED10):
        raise TypeError("spectrum must be a BinnedSED10 instance.")

    if atol < 0.0:
        raise ValueError("atol must be non-negative.")

    if not hasattr(source_response, "expectation") or not hasattr(source_response, "_source"):
        raise TypeError(
            "source_response must be a BinnedThreeMLExtendedSourceResponse-like object."
        )

    if source_response._source is None:
        raise RuntimeError("Call source_response.set_source(source) first.")

    attached_spectrum = source_response._source.spectrum.main.shape
    if attached_spectrum is not spectrum:
        raise ValueError(
            "The supplied spectrum is not the spectrum attached to source_response."
        )

    parameters = [getattr(spectrum, f"K{i}") for i in range(10)]
    saved_values = [float(par.value) for par in parameters]
    saved_free = [bool(par.free) for par in parameters]

    totals = np.zeros(10, dtype=float)

    try:
        for i, par_i in enumerate(parameters):
            for par in parameters:
                par.value = 0.0

            par_i.value = _positive_test_value(par_i)

            expectation = source_response.expectation(copy=False)
            template = _dense_contents(expectation)
            totals[i] = float(np.sum(np.abs(template)))

    finally:
        for par, value, free in zip(parameters, saved_values, saved_free):
            par.value = value
            par.free = free
    zero_local = np.where(totals <= atol)[0]

    response_indices = np.asarray(
        getattr(spectrum, "_cosipy_ei_bin_indices", tuple(range(10))),
        dtype=int,
    )
    zero_response = response_indices[zero_local]

    if freeze and zero_local.size:
        freeze_binned_sed10_bins(spectrum, zero_local)

        warnings.warn(
            "The following BinnedSED10 bins produce zero source counts in the "
            "selected detector analysis space and were frozen to K=0: "
            f"local={zero_local.tolist()}, Ei={zero_response.tolist()}."
        )

    if verbose:
        for i in range(10):
            state = "ZERO" if i in set(zero_local.tolist()) else "ok"
            print(
                f"SED bin {i:2d} (Ei {response_indices[i]:2d}): "
                f"template counts={totals[i]:.6g}  [{state}]"
            )

    return {
        "local_indices": zero_local,
        "response_indices": zero_response,
        "template_totals": totals,
    }


def profile_likelihood_upper_limit(
    profile_nll,
    nll_best,
    best_value,
    sigma=None,
    delta_ts=2.705543,
    max_value=None,
    max_bracket_steps=60,
    rtol=1e-5,
):
    """
    Compute a one-sided profile-likelihood upper limit.

    ``profile_nll(value)`` must evaluate the negative log likelihood with the
    parameter of interest fixed to ``value`` while re-optimizing all desired
    nuisance parameters. The returned limit solves

        2 * [NLL_profile(value) - NLL_best] = delta_ts

    with ``delta_ts=2.705543`` corresponding to the usual one-sided 95%
    profile-likelihood threshold for one parameter.

    Parameters
    ----------
    profile_nll : callable
        Function of one non-negative parameter value returning the profiled NLL.
    nll_best : float
        Global best-fit negative log likelihood.
    best_value : float
        Best-fit value of the parameter of interest.
    sigma : float, optional
        Approximate 1-sigma uncertainty, used only to choose the initial upper
        bracket. If unavailable, a scale is inferred from ``best_value``.
    delta_ts : float, optional
        Target value of ``2 Delta NLL``. Default is 2.705543.
    max_value : float, optional
        Hard maximum allowed for the parameter.
    max_bracket_steps : int, optional
        Maximum number of bracket expansions.
    rtol : float, optional
        Relative tolerance passed to ``scipy.optimize.brentq``.

    Returns
    -------
    float
        Upper limit, or NaN if the target could not be bracketed.
    """

    nll_best = float(nll_best)
    best_value = max(float(best_value), 0.0)
    delta_ts = float(delta_ts)

    if not np.isfinite(nll_best):
        raise ValueError("nll_best must be finite.")
    if not np.isfinite(best_value):
        raise ValueError("best_value must be finite.")
    if delta_ts <= 0.0 or not np.isfinite(delta_ts):
        raise ValueError("delta_ts must be finite and positive.")

    if sigma is None or not np.isfinite(sigma) or sigma <= 0.0:
        sigma = max(abs(best_value), 1e-12)
    else:
        sigma = float(sigma)

    if max_value is not None:
        max_value = float(max_value)
        if max_value <= best_value:
            raise ValueError("max_value must be greater than best_value.")

    def root_function(value):
        value = max(float(value), 0.0)
        profiled = float(profile_nll(value))
        return 2.0 * (profiled - nll_best) - delta_ts

    lower = best_value
    upper = max(
        lower + 3.0 * sigma,
        2.0 * lower,
        10.0 * sigma,
    )

    if max_value is not None:
        upper = min(upper, max_value)

    for _ in range(int(max_bracket_steps)):
        f_upper = root_function(upper)

        if np.isfinite(f_upper) and f_upper >= 0.0:
            return float(
                brentq(
                    root_function,
                    lower,
                    upper,
                    rtol=rtol,
                    maxiter=100,
                )
            )

        if max_value is not None and upper >= max_value:
            break

        upper *= 2.0
        if max_value is not None:
            upper = min(upper, max_value)

    warnings.warn(
        "Could not bracket the requested profile-likelihood upper limit. "
        "Returning NaN."
    )
    return np.nan
