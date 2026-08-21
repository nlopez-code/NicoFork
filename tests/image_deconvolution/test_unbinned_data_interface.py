"""Synthetic consistency test for UnbinnedImageDataInterface.

Mock IRF: A_eff = 10 cm^2 everywhere; event density P uniform over the data
space (Em range 12 keV, phi range pi, psichi 4pi sr) so P = 1/(12*pi*4pi).
SC attitude: slow rotation about z (galactic frame), 1 s livetime bins.

Analytic expectations:
  exposure_j     = A_eff * T_live * pixarea            (all pixels equal)
  with flux c = n_events / (4pi * A_eff * T_live):
     lambda_i    = c * pixarea * n_sky * A_eff * P = c * 4pi * A_eff * P
     T_product_j = pixarea * n_ev * A_eff * P / lambda_i = pixarea*n_ev/(c*4pi)
     ratio_j     = T_product_j / exposure_j = n_ev / (c*4pi*A_eff*T) = 1
"""
import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import GCRS
from scoords import Attitude
from scipy.spatial.transform import Rotation

from cosipy.spacecraftfile.spacecraft_file import SpacecraftHistory
from cosipy.image_deconvolution.unbinned_image_data_interface import (
    UnbinnedImageDataInterface,
)

AEFF = 10.0
P_UNIFORM = 1.0 / (12.0 * np.pi * 4.0 * np.pi)


class MockIRF:
    def event_probability(self, photons, events):
        n = len(np.asarray(photons.direction_lon_rad_sc))
        return np.full(n, P_UNIFORM)

    def _effective_area_cm2(self, photons):
        n = len(np.asarray(photons.direction_lon_rad_sc))
        return np.full(n, AEFF)


class MockEvents:
    def __init__(self, times, rng, n):
        self.nevents = n
        self.jd1 = times.jd1
        self.jd2 = times.jd2
        self.energy_keV = np.full(n, 511.0)
        self.scattering_angle_rad = rng.uniform(0.1, 2.0, n)
        self.scattered_lon_rad_sc = rng.uniform(0, 2 * np.pi, n)
        self.scattered_lat_rad_sc = rng.uniform(-1.0, 1.0, n)


def test_unbinned_interface_consistency():
    rng = np.random.default_rng(42)

    # --- SC history: 101 timestamps, rotation about z, 1 s bins ---
    n_pts = 101
    t0 = Time("2026-01-01T00:00:00")
    obstime = t0 + np.arange(n_pts) * u.s
    angles = np.linspace(0, 0.5, n_pts)  # rad, slow rotation
    att = Attitude(Rotation.from_euler("z", angles[:, None]), frame="galactic")
    loc = GCRS(ra=0 * u.deg, dec=0 * u.deg, distance=7000 * u.km)
    sc = SpacecraftHistory(obstime, att, loc)
    T_live = float(np.sum(sc.livetime.to(u.s).value))

    # --- events strictly inside the history ---
    n_ev = 50
    ev_times = t0 + rng.uniform(1, n_pts - 2, n_ev) * u.s
    events = MockEvents(ev_times, rng, n_ev)

    nside = 2
    interface = UnbinnedImageDataInterface(
        irf=MockIRF(),
        events=events,
        nside=nside,
        sc_history=sc,
        energy_edges=[505.0, 517.0],
        background_models={"flat_bkg": np.full(n_ev, 7.0)},
    )

    R = interface.response_matrix
    n_sky = 12 * nside**2
    pixarea = 4 * np.pi / n_sky

    # 1. shape
    assert R.shape == (n_sky + 1, n_ev), R.shape
    print(f"1. response shape {R.shape} OK")

    # 2. sky rows = A_eff * P
    assert np.allclose(R[:n_sky], AEFF * P_UNIFORM)
    print("2. sky rows = A_eff * P OK")

    # 3. exposure = A_eff * T_live * pixarea
    expo = interface.exposure_map.contents[:, 0]
    assert np.allclose(expo, AEFF * T_live * pixarea, rtol=1e-10), (
        expo[0], AEFF * T_live * pixarea)
    print(f"3. exposure = A_eff*T*pixarea = {expo[0]:.6f} cm2 s sr OK")

    # 4. background row kept at absolute scale; summed model = total
    assert np.allclose(R[n_sky], 7.0)
    assert np.isclose(interface.summed_bkg_model("flat_bkg"), 7.0 * n_ev)
    print("4. bkg row absolute scale + summed_bkg_model = total OK")

    # 5. RL fixed point: flux c with N_expected = n_ev -> T_product/exposure = 1
    c = n_ev / (4 * np.pi * AEFF * T_live)
    model = np.full(n_sky, c)
    lam = interface.calc_source_expectation(model).contents
    expected_lam = c * 4 * np.pi * AEFF * P_UNIFORM
    assert np.allclose(lam, expected_lam), (lam[0], expected_lam)
    ratio = interface.calc_T_product(1.0 / lam).contents[:, 0] / expo
    assert np.allclose(ratio, 1.0, rtol=1e-10), ratio[:3]
    print("5. RL fixed-point ratio T_product/exposure = 1 exactly OK")

    # 6. total expected counts: sum_j m_j * exposure_j = n_ev
    assert np.isclose(np.sum(model * expo), n_ev)
    print("6. sum(model * exposure) = n_events OK")

    # 7. log-likelihood form
    ll = interface.calc_log_likelihood(interface.calc_source_expectation(model))
    assert np.isclose(ll, n_ev * np.log(expected_lam) - n_ev)
    print("7. logL = sum(ln lambda) - n_events OK")

    # 8. multi-energy-bin guard
    try:
        UnbinnedImageDataInterface(
            irf=MockIRF(), events=events, nside=nside, sc_history=sc,
            energy_edges=[505.0, 511.0, 517.0])
        raise AssertionError("guard did not fire")
    except NotImplementedError:
        print("8. multi-energy-bin guard raises NotImplementedError OK")

    # 9. livetime-weighted downsampling preserves total livetime
    expo_ds = interface._compute_exposure_from_sc(n_time_samples=10)
    assert np.allclose(expo_ds, AEFF * T_live * pixarea, rtol=1e-10)
    print("9. downsampled exposure preserves total livetime OK")

    print("\nAll consistency checks passed.")
