import numpy as np
import healpy as hp
import logging
from astropy import units as u
from astropy.time import Time
from histpy import Histogram, Axes, Axis, HealpixAxis
from cosipy.response.photon_types import PhotonListWithDirectionAndEnergyInSCFrame
from cosipy.background_estimation import BinnedBackgroundRates
from cosipy.image_deconvolution.data_interfaces.image_deconvolution_data_interface_base import (
    ImageDeconvolutionDataInterfaceBase,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
2

class UnbinnedImageDataInterface(ImageDeconvolutionDataInterfaceBase):
    """
    Unbinned image data interface for RL deconvolution.

    The full response matrix has shape (n_sky + n_bkg, n_events):
      - Rows 0 .. n_sky-1          : A_eff(j_SC(t_i)) * P(event_i | photon from sky pixel j),
                                     where P is a density per (dEm x dΦ x dΩ_psichi)
      - Rows n_sky .. n_sky+n_bkg-1: per-event background density b(x_i) in the
                                     same data-space measure, kept at its absolute
                                     (expected-counts) scale.  A background
                                     normalization of 1.0 therefore corresponds to
                                     the model-predicted counts, and the summed
                                     background model is the total expected counts.

    The exposure map holds A_eff_j * Δt * ΔΩ_pix (cm^2 s sr) per sky pixel,
    computed by integrating the effective area over the spacecraft history.
    The pixel solid angle is included to match the pixarea factor applied in
    calc_source_expectation / calc_T_product for per-steradian model fluxes.
    """

    def __init__(
        self,
        irf,
        events,
        nside,
        sc_history,
        energy_edges,
        exposure_map=None,
        background_models=None,
    ):
        """
        Parameters
        ----------
        irf : instrument response function
        events : EmCDSEventDataInSCFrameInterface
            Time-tagged event list in SC frame.
        nside : int
            HEALPix nside for the galactic model map (RING scheme).
        sc_history : cosipy.spacecraftfile.SpacecraftHistory
            Spacecraft history used to interpolate the attitude at each
            event's timestamp and to compute the exposure map.
        energy_edges : array-like
            Edges in keV of the reconstructed model's incident-energy axis.
            This discretizes the *sky model* (which RL reconstructs on a
            grid), not the event data: each event's measured energy Em enters
            the likelihood unbinned.  The arithmetic midpoint of each bin,
            ``(E_lo + E_hi) / 2``, is used as the incident photon energy Ei
            at which the IRF (response density and effective area) is
            evaluated.  There will always be ``len(energy_edges) - 1`` bins.
        exposure_map : array-like, optional
            Pre-computed per-pixel normalization of shape (npix,), in units
            of cm^2 s sr (A_eff × livetime × pixel solid angle).  When
            omitted the normalization is derived from A_eff × livetime ×
            ΔΩ_pix using the spacecraft history.
        background_models : dict, optional
            Maps background model name → per-event rates.  Each value may be:
              - A 1-D array of shape (n_events,) already containing b(Ω_i)
                densities.
              - A BinnedBackgroundRates object (from
                background_rates_from_binned_estimate).
              - Any object that implements expectation_density() (e.g.
                FreeNormNFUnbinnedBackground).
        """
        super().__init__()

        self._irf = irf
        self._nside = nside
        self._npix = hp.nside2npix(self._nside)
        self._pixarea = hp.nside2pixarea(self._nside, degrees=False)  # sr
        self._events = events
        self._n_events = events.nevents
        self._name = "UnbinnedImageDataInterface"
        self._energy_arr = np.asarray(events.energy_keV, dtype=float)

        # --- Model incident-energy axis (Ei) ---
        # This discretizes the reconstructed sky model, not the event data.
        # Each event's measured energy Em (self._energy_arr) is used unbinned
        # by the IRF.
        self._energy_edges_keV = np.asarray(energy_edges, dtype=float)
        self._energy_mids_keV = 0.5 * (
            self._energy_edges_keV[:-1] + self._energy_edges_keV[1:]
        )
        self._n_energy_bins = len(self._energy_mids_keV)
        if self._n_energy_bins != 1:
            raise NotImplementedError(
                "UnbinnedImageDataInterface currently supports a single model "
                f"incident-energy bin (got {self._n_energy_bins}). The response "
                "matrix has no incident-energy axis: _coerce_model, "
                "calc_T_product and the exposure map only address bin 0."
            )
        # Single incident energy Ei (the bin's arithmetic midpoint) for every event's
        # photon-list context.  Per-event array so its length matches the IRF
        # event list; all entries equal the midpoint in the single-bin case.
        self._incident_energy_arr = np.full(
            self._n_events, self._energy_mids_keV[0], dtype=float
        )

        # --- Resolve per-event rotation matrices (galactic → SC frame) ---
        event_times = Time(events.jd1, events.jd2, format='jd')
        att_at_events = sc_history.interp_attitude(event_times)
        self._rot_mats_gal_to_sc = att_at_events.rot.inv().as_matrix()  # (n_events, 3, 3)

        # Store sc_history for exposure map computation.
        self._sc_history = sc_history

        # All HEALPix pixels in galactic coordinates (RING scheme).
        self._pix_array = np.arange(self._npix)
        self._n_sky = self._npix
        self._n_model = self._npix

        thetas, phis = hp.pix2ang(self._nside, self._pix_array)
        self._gal_vecs = hp.ang2vec(thetas, phis)  # (n_sky, 3)

        # Model axes: full-sky 2D (HealpixAxis + EnergyAxis)
        image_axis  = HealpixAxis(nside=self._nside, scheme='ring', coordsys='galactic', label='lb')
        energy_axis = Axis(edges=u.Quantity(self._energy_edges_keV, u.keV), label='Ei', scale='log')
        self._model_axes = Axes([image_axis, energy_axis])
        self._data_axes  = Axes([Axis(np.arange(self._n_events + 1), label='event')])

        self._event = Histogram(self._data_axes, contents=np.ones(self._n_events))

        # --- Pre-computed sky exposure map (optional) ---
        if exposure_map is not None:
            aeff_arr = np.asarray(exposure_map, dtype=float)
            if aeff_arr.shape != (self._n_sky,):
                raise ValueError(
                    f"exposure map must have shape ({self._n_sky},), "
                    f"got {aeff_arr.shape}"
                )
            aeff_fullsky = np.zeros((self._npix, 1), dtype=float)
            aeff_fullsky[self._pix_array, 0] = aeff_arr
            self._exposure_map = Histogram(self._model_axes, contents=aeff_fullsky)
        else:
            self._exposure_map = None  # built lazily from SC history

        # --- Background models ---
        if background_models is None:
            background_models = {}

        self._bkg_names = list(background_models.keys())
        self._n_bkg = len(self._bkg_names)
        self._bkg_models = {}
        self._summed_bkg_models = {}
        self._bkg_density_interfaces = {}

        for key, values in background_models.items():
            if hasattr(values, 'expectation_density'):
                self._bkg_density_interfaces[key] = values
                arr = np.asarray(values.expectation_density(), dtype=float)
                total = (float(values.expected_counts())
                         if hasattr(values, 'expected_counts')
                         else float(np.sum(arr)))
            elif isinstance(values, BinnedBackgroundRates):
                arr = np.asarray(values.per_event_density, dtype=float)
                total = values.total_rate
            else:
                arr = np.asarray(values, dtype=float)
                total = float(np.sum(arr))

            if arr.shape != (self._n_events,):
                raise ValueError(
                    f"background model '{key}' must have shape ({self._n_events},), "
                    f"got {arr.shape}"
                )

            # Keep the absolute expected-counts density b(x_i) so that a
            # background normalization of 1.0 corresponds to the
            # model-predicted counts (the cosipy convention; works with both
            # RLbasic, where the norm stays fixed at 1.0, and RL, where it is
            # fitted within a range around 1).  The RL M-step denominator is
            # the integral of the row over the data space, i.e. the total
            # expected counts.  In ToyUnbinnedRL the row is instead
            # unit-normalised with the norm fitted to the total count; the
            # two parametrizations are equivalent up to this scale.
            self._bkg_models[key] = Histogram(self._data_axes, contents=arr)
            self._summed_bkg_models[key] = total

        self._response_matrix = None  # built lazily; shape (n_sky + n_bkg, n_events)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pix_array(self):
        return self._pix_array

    @property
    def energy_edges(self):
        return self._model_axes['Ei'].edges

    @property
    def exposure_map(self):
        if self._exposure_map is None:
            _ = self.response_matrix  # triggers build which sets _exposure_map
        return self._exposure_map

    @property
    def response_matrix(self):
        """Full response matrix of shape (n_sky + n_bkg, n_events).

        Sky rows (0 .. n_sky-1) hold A_eff * P(event | sky pixel).
        Background rows (n_sky .. n_sky+n_bkg-1) hold per-event densities.
        """
        if self._response_matrix is None:
            self._response_matrix = self._build_response_matrix()
        return self._response_matrix

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build_response_matrix(self):
        has_aeff = hasattr(self._irf, '_effective_area_cm2')
        if not has_aeff:
            logger.warning(
                "IRF does not expose _effective_area_cm2; sky rows will contain "
                "probability densities without A_eff scaling. Provide a pre-computed "
                "exposure_map for accurate results."
            )

        sky_matrix = np.zeros((self._n_sky, self._n_events), dtype=float)

        for i, v_gal in enumerate(self._gal_vecs):
            v_sc = self._rot_mats_gal_to_sc @ v_gal  # (n_events, 3)

            # The (phi, arm, az) transformation is performed internally by the
            # NF response (NFResponseModels._convert_conventions); only the
            # source direction in the SC frame is needed here.
            lon_sc = np.arctan2(v_sc[:, 1], v_sc[:, 0])
            lat_sc = np.pi / 2.0 - np.arccos(np.clip(v_sc[:, 2], -1.0, 1.0))

            photon_list = PhotonListWithDirectionAndEnergyInSCFrame(
                lon_sc, lat_sc, self._incident_energy_arr
            )
            probs = self._irf.event_probability(photon_list, self._events)
            prob_arr = np.fromiter(probs, dtype=float, count=self._n_events)

            if has_aeff:
                # R(j; Ω_i) = A_eff(j_SC(t_i), E_i) * P_j(Ω_i)
                aeff = np.asarray(self._irf._effective_area_cm2(photon_list), dtype=float)
                sky_matrix[i, :] = prob_arr * aeff
            else:
                sky_matrix[i, :] = prob_arr

            if (i + 1) % 10 == 0 or (i + 1) == self._n_sky:
                logger.info(
                    "Response matrix (sky): %.1f%% complete.",
                    100.0 * (i + 1) / self._n_sky,
                )

        # --- Exposure map: A_eff_j * Δt, integrated over spacecraft history ---
        if self._exposure_map is None:
            if has_aeff and self._sc_history is not None:
                norm = self._compute_exposure_from_sc()
                logger.info("Computed exposure map from spacecraft history (A_eff × livetime).")
            else:
                logger.warning(
                    "Falling back to sum-over-events exposure approximation. "
                    "Pass a pre-computed exposure_map for accurate results."
                )
                norm = np.sum(sky_matrix, axis=1) * self._pixarea
            norm_fullsky = np.zeros((self._npix, 1), dtype=float)
            norm_fullsky[self._pix_array, 0] = norm
            self._exposure_map = Histogram(self._model_axes, contents=norm_fullsky)

        # --- Background block ---
        if self._n_bkg > 0:
            bkg_rows = np.stack(
                [
                    self._bkg_models[k].contents
                    if hasattr(self._bkg_models[k], "contents")
                    else self._bkg_models[k]
                    for k in self._bkg_names
                ],
                axis=0,
            )  # (n_bkg, n_events)
            full_matrix = np.vstack([sky_matrix, bkg_rows])
            logger.info("Appended %d background row(s) to response matrix.", self._n_bkg)
        else:
            full_matrix = sky_matrix

        return full_matrix  # (n_sky + n_bkg, n_events)

    # ------------------------------------------------------------------

    def _compute_exposure_from_sc(
        self, energy_keV: float = None, n_time_samples: int = 5000
    ) -> np.ndarray:
        """Integrate A_eff over the spacecraft history to obtain the exposure map.

        Parameters
        ----------
        energy_keV : float, optional
            Representative photon energy for A_eff evaluation.
            Defaults to the mean event energy.
        n_time_samples : int, optional
            Maximum number of SC history time steps to use. The history is
            down-sampled evenly if it has more steps. Default 5000.

        Returns
        -------
        np.ndarray, shape (n_sky,)
            Exposure per sky pixel in cm^2 * s * sr.  The pixel solid angle
            is included so that the RL M-step denominator matches the
            pixarea factor applied in calc_T_product / calc_source_expectation
            (model fluxes are per steradian; same convention as
            dataIF_COSI_DC2._calc_exposure_map).
        """
        if energy_keV is None:
            energy_keV = float(np.mean(self._energy_mids_keV))

        sc = self._sc_history
        # Rotation matrices at each SC history point: (n_points, 3, 3)
        rot_mats = sc.attitude.rot.inv().as_matrix()
        livetime = sc.livetime.to(u.s).value  # (n_intervals,) = (n_points - 1,)
        n_intervals = len(livetime)

        # Associate each interval with its start-of-interval rotation.
        rot_intervals = rot_mats[:-1]  # (n_intervals, 3, 3)

        if n_intervals > n_time_samples:
            # Split the history into contiguous chunks; weight each sampled
            # attitude by the summed livetime of its chunk.  This preserves
            # the total livetime exactly and keeps dead-time structure
            # (e.g. SAA passes) from being smeared uniformly over the orbit.
            chunk_edges = np.linspace(0, n_intervals, n_time_samples + 1, dtype=int)
            cum_lt = np.concatenate(([0.0], np.cumsum(livetime)))
            lt_sampled = cum_lt[chunk_edges[1:]] - cum_lt[chunk_edges[:-1]]
            chunk_centers = (chunk_edges[:-1] + chunk_edges[1:]) // 2
            rot_sampled = rot_intervals[chunk_centers]
        else:
            rot_sampled = rot_intervals
            lt_sampled = livetime

        n_t = len(lt_sampled)

        # Rotate all sky pixels into SC frame for every sampled time step.
        # v_sc_all[t, j, :] = rot_sampled[t] @ gal_vecs[j]
        v_sc_all = np.einsum('tij,sj->tsi', rot_sampled, self._gal_vecs)  # (n_t, n_sky, 3)
        v_sc_flat = v_sc_all.reshape(n_t * self._n_sky, 3)

        lon_sc = np.arctan2(v_sc_flat[:, 1], v_sc_flat[:, 0])
        lat_sc = np.pi / 2.0 - np.arccos(np.clip(v_sc_flat[:, 2], -1.0, 1.0))
        energy_flat = np.full(n_t * self._n_sky, energy_keV, dtype=float)

        photon_list = PhotonListWithDirectionAndEnergyInSCFrame(
            lon_sc, lat_sc, energy_flat
        )
        aeff_flat = np.asarray(self._irf._effective_area_cm2(photon_list), dtype=float)

        aeff_all = aeff_flat.reshape(n_t, self._n_sky)   # (n_t, n_sky)
        exposure = (aeff_all * lt_sampled[:, np.newaxis]).sum(axis=0)  # (n_sky,) cm^2·s
        return exposure * self._pixarea  # cm^2·s·sr

    # ------------------------------------------------------------------
    # Interface methods
    # ------------------------------------------------------------------

    def _coerce_model(self, model):
        if isinstance(model, Histogram):
            contents = model.contents
            if hasattr(contents, 'value'):
                contents = contents.value
            arr = np.asarray(contents[self._pix_array, 0], dtype=float).flatten()
        else:
            arr = np.asarray(model, dtype=float).flatten()
        if arr.shape != (self._n_sky,):
            raise ValueError(
                f"model must have {self._n_sky} elements, got {arr.shape}"
            )
        return arr

    _ALMOST_ZERO = 1e-12

    def calc_source_expectation(self, model):
        arr = self._coerce_model(model)
        result = self.response_matrix[:self._n_sky, :].T @ arr * self._pixarea
        result = np.where(result < self._ALMOST_ZERO, self._ALMOST_ZERO, result)
        return Histogram(self._data_axes, contents=result)

    def calc_bkg_expectation(self, dict_bkg_norm=None):
        expectation = np.zeros(self._n_events, dtype=float)
        for i, key in enumerate(self._bkg_names):
            norm = dict_bkg_norm.get(key, 1.0) if dict_bkg_norm else 1.0
            expectation += norm * self.response_matrix[self._n_sky + i, :]
        return Histogram(self._data_axes, contents=expectation)

    def calc_expectation(self, model, dict_bkg_norm=None):
        combined = (
            self.calc_source_expectation(model).contents
            + self.calc_bkg_expectation(dict_bkg_norm).contents
        )
        return Histogram(self._data_axes, contents=combined)

    def calc_T_product(self, dataspace_histogram):
        if isinstance(dataspace_histogram, Histogram):
            data_vec = dataspace_histogram.contents.flatten()
        else:
            data_vec = np.asarray(dataspace_histogram, dtype=float).flatten()
        result_partial = self.response_matrix[:self._n_sky, :] @ data_vec * self._pixarea
        result_fullsky = np.zeros((self._npix, 1), dtype=float)
        result_fullsky[self._pix_array, 0] = result_partial
        return Histogram(self._model_axes, contents=result_fullsky)

    def calc_bkg_model_product(self, key, dataspace_histogram):
        if isinstance(dataspace_histogram, Histogram):
            data_vec = dataspace_histogram.contents.flatten()
        else:
            data_vec = np.asarray(dataspace_histogram, dtype=float).flatten()
        i = self._bkg_names.index(key)
        return float(np.dot(self.response_matrix[self._n_sky + i, :], data_vec))

    def calc_expected_counts(self, model, dict_bkg_norm=None):
        """Total expected counts <N_tot>, the integral of lambda over the data space.

        Summing the per-event densities lambda_i does NOT give this integral.
        It instead factorizes over the model, because P(x | sky pixel j) is
        normalized over the data space:

            <N_tot> = sum_j exposure_j * m_j  +  sum_k norm_k * total_bkg_k

        The sky term uses the exposure map (A_eff * livetime * dOmega_pix),
        which is also the RL M-step denominator.  The background totals are
        already absolute expected counts, so a norm of 1.0 contributes the
        model-predicted counts.
        """
        arr = self._coerce_model(model)

        exposure = self.exposure_map.contents
        if hasattr(exposure, 'value'):
            exposure = exposure.value
        exposure = np.asarray(exposure, dtype=float)[self._pix_array, 0]

        n_tot = float(np.dot(exposure, arr))
        for key in self._bkg_names:
            norm = dict_bkg_norm.get(key, 1.0) if dict_bkg_norm else 1.0
            n_tot += norm * self._summed_bkg_models[key]
        return n_tot

    def calc_log_likelihood(self, expectation, model=None, dict_bkg_norm=None):
        """Unbinned Poisson log-likelihood: -<N_tot> + sum_i ln(lambda_i).

        lambda_i = <N_tot> * Prob(x_i) is the per-event intensity density held
        in ``expectation``, so only the integral term needs the model.

        When ``model`` is omitted <N_tot> falls back to n_events.  That is exact
        only at the RL fixed point, where counts are conserved; away from it the
        error varies from iteration to iteration, so likelihood *differences* --
        which drive the stopping criterion and the accelerator's accept/reject
        test -- come out wrong.  Callers holding the model should pass it.
        """
        if isinstance(expectation, Histogram):
            arr = expectation.contents.flatten()
        else:
            arr = np.asarray(expectation, dtype=float).flatten()
        arr = np.where(arr <= 0, self._ALMOST_ZERO, arr)

        n_tot = self._n_events if model is None else self.calc_expected_counts(model, dict_bkg_norm)
        return float(np.sum(np.log(arr)) - n_tot)

    # ------------------------------------------------------------------
    # Background utilities
    # ------------------------------------------------------------------

    def background_rates_from_sky_model(self, sky_flux):
        R   = self.response_matrix[:self._n_sky, :]  # (n_sky, n_events)
        arr = np.asarray(sky_flux, dtype=float)
        if arr.ndim == 1:
            return R.T @ arr
        return np.sum(R * arr, axis=0)

    def add_background_model(self, name, rates):
        """Add (or replace) a background model after construction.

        Parameters
        ----------
        name : str
        rates : array-like or BinnedBackgroundRates
            Per-event background densities b(Ω_i).
        """
        if isinstance(rates, BinnedBackgroundRates):
            arr = np.asarray(rates.per_event_density, dtype=float)
            total = rates.total_rate
        else:
            arr = np.asarray(rates, dtype=float)
            total = float(np.sum(arr))

        if arr.shape != (self._n_events,):
            raise ValueError(
                f"Background model '{name}' must have shape ({self._n_events},), "
                f"got {arr.shape}"
            )

        if name in self._bkg_names:
            i = self._bkg_names.index(name)
            self._bkg_models[name]        = Histogram(self._data_axes, contents=arr)
            self._summed_bkg_models[name] = total
            if self._response_matrix is not None:
                self._response_matrix[self._n_sky + i, :] = arr
        else:
            self._bkg_names.append(name)
            self._n_bkg += 1
            self._bkg_models[name]        = Histogram(self._data_axes, contents=arr)
            self._summed_bkg_models[name] = total
            if self._response_matrix is not None:
                self._response_matrix = np.vstack(
                    [self._response_matrix, arr.reshape(1, -1)]
                )

    @staticmethod
    def background_rates_from_binned_estimate(estimated_bg, events):
        """Convert a binned background histogram to per-event densities.

        Looks up b_i (the bin count) for each event and divides by the
        data-space phase-space volume ΔΩ_i = ΔEm * ΔΦ * Ω_pixel to obtain
        the probability density b(Ω_i) = b_i / ΔΩ_i required for the
        unbinned RL algorithm.

        Parameters
        ----------
        estimated_bg : histpy.Histogram
            Binned background estimate with axes Em, Phi, PsiChi.
        events : EmCDSEventDataInSCFrameInterface

        Returns
        -------
        BinnedBackgroundRates
            Holds per_event_density (shape n_events) and total_rate
            (sum of all bins, used as RL background-norm denominator).
        """
        energy = np.asarray(events.energy_keV,           dtype=float)
        phi    = np.asarray(events.scattering_angle_rad, dtype=float)
        lon_sc = np.asarray(events.scattered_lon_rad_sc, dtype=float)
        lat_sc = np.asarray(events.scattered_lat_rad_sc, dtype=float)

        # Em bin indices
        em_edges = estimated_bg.axes['Em'].edges
        if hasattr(em_edges, 'value'):
            em_edges = em_edges.to('keV').value
        em_idx = np.clip(
            np.searchsorted(em_edges, energy, side='right') - 1,
            0, estimated_bg.axes['Em'].nbins - 1,
        )

        # Phi bin indices
        phi_edges = estimated_bg.axes['Phi'].edges
        if hasattr(phi_edges, 'value'):
            phi_edges = phi_edges.to('rad').value
        phi_idx = np.clip(
            np.searchsorted(phi_edges, phi, side='right') - 1,
            0, estimated_bg.axes['Phi'].nbins - 1,
        )

        # PsiChi HEALPix pixel
        psichi_axis = estimated_bg.axes['PsiChi']
        theta_sc    = np.pi / 2.0 - lat_sc
        psichi_pix  = psichi_axis.ang2pix(theta_sc, lon_sc)

        # Look up bin counts
        contents = estimated_bg.contents
        if hasattr(contents, 'todense'):
            contents = contents.todense()
        if hasattr(contents, 'value'):
            contents = contents.value
        contents = np.asarray(contents, dtype=float)

        raw_counts = contents[em_idx, phi_idx, psichi_pix]

        # Phase-space volume ΔΩ_i = ΔEm * ΔΦ * Ω_pixel
        em_widths  = np.diff(em_edges)
        phi_widths = np.diff(phi_edges)
        delta_em   = em_widths[em_idx]
        delta_phi  = phi_widths[phi_idx]
        omega_pixel = hp.nside2pixarea(psichi_axis.nside, degrees=False)  # sr

        density = raw_counts / (delta_em * delta_phi * omega_pixel)

        # Total expected background = sum of all bin counts (RL normalization)
        total_rate = float(np.sum(contents))

        return BinnedBackgroundRates(density, total_rate)