import numpy as np
import healpy as hp
import mhealpy as mp
import logging
from astropy import units as u
from histpy import Histogram
from scoords.spacecraft_frame import SpacecraftFrame
from astropy.coordinates import SkyCoord
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame
from cosipy.image_deconvolution.image_deconvolution_data_interface_base import (
    ImageDeconvolutionDataInterfaceBase,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class UnbinnedImageDataInterface(ImageDeconvolutionDataInterfaceBase):
    """
    Unbinned image data interface for RL-style deconvolution.

    response_matrix[j, i] = P(event_i | photon_j)
    exposure_map[j]       = full-phase-space normalization for model pixel j
    """

    def __init__(
        self,
        irf,
        events,
        nside,
        radius_deg,
        exposure_map=None,
        background_models=None,
        source_dir=None,
        energy = 1050 * u.keV
    ):
        super().__init__()

        self._irf = irf
        self._nside = nside
        self._npix = hp.nside2npix(self._nside)

        self._events = events

        self._n_events = events.nevents
        self._event = 1.0
        self._name = "UnbinnedImageDataInterface"
        self._exposure_map = None
        self._bkg_models = None
        self._summed_bkg_models = None
        self._energy = energy
        self._radius_deg = radius_deg

        # Define model-space pixels
        if source_dir is None:
            source_dir = SkyCoord(lon=0.0, lat=75.0, unit="deg", frame=SpacecraftFrame())

        lon = source_dir.lon.to_value(u.rad)
        lat = source_dir.lat.to_value(u.rad)

        theta = 0.5 * np.pi - lat
        phi = lon

        vec = hp.ang2vec(theta, phi)
        
        if radius_deg == None:
            self._pix_array = mp.nside2npix(self._nside)
        else:
            radius_rad = np.deg2rad(radius_deg)
            self._pix_array = mp.query_disc(self._nside, vec, radius_rad)

        # Build photons and effective area
        self._photons = []
        aeff = np.zeros(len(self._pix_array), dtype=float)

        for i, pix in enumerate(self._pix_array):
            theta_i, phi_i = hp.pix2ang(self._nside, pix)

            photon = PhotonWithDirectionAndEnergyInSCFrame(
                phi_i,
                0.5 * np.pi - theta_i,
                energy.to_value(u.keV),
            )

            self._photons.append(photon)
            aeff[i] = self._irf.effective_area_cm2(photon)

        self._n_model = len(self._photons)

        if exposure_map is None:
            self._exposure_map = aeff.copy()
        else:
            self._exposure_map = np.asarray(exposure_map, dtype=float)
            if self._exposure_map.shape != (self._n_model,):
                raise ValueError(
                    f"exposure_map must have shape ({self._n_model},), "
                    f"got {self._exposure_map.shape}"
                )

        if background_models is None:
            background_models = {"background": np.zeros(self._n_events, dtype=float)}

        self._bkg_models = {}
        self._summed_bkg_models = {}

        for key, values in background_models.items():
            arr = np.asarray(values, dtype=float)
            if arr.shape != (self._n_events,):
                raise ValueError(
                    f"background model '{key}' must have shape ({self._n_events},), "
                    f"got {arr.shape}"
                )
            self._bkg_models[key] = arr
            self._summed_bkg_models[key] = None

        self._response_matrix = None

    @property
    def pix_array(self):
        return self._pix_array

    @property
    def response_matrix(self):
        if self._response_matrix is None:
            self._response_matrix = self._build_response_matrix()
        return self._response_matrix

    def _build_response_matrix(self):
        prob_matrix = np.zeros((self._n_model, self._n_events), dtype=float)

        for i, photon in enumerate(self._photons):
            probs = self._irf.event_probability(photon, self._events)
            prob_matrix[i, :] = np.asarray(list(probs), dtype=float)

            if (i + 1) % 10 == 0 or (i + 1) == self._n_model:
                logger.info(
                    "Computed %d / %d rows of probability matrix.",
                    i + 1,
                    self._n_model,
                )

        return prob_matrix

    def _coerce_model(self, model):
        arr = np.asarray(model, dtype=float)
        if arr.shape != (self._n_model,):
            raise ValueError(
                f"model must have shape ({self._n_model},), got {arr.shape}"
            )
        return arr

    def calc_expectation(self, model, dict_bkg_norm=None, almost_zero=1e-12):
        model = self._coerce_model(model)
        expectation = self.response_matrix.T @ model

        if dict_bkg_norm is None:
            for key in self.keys_bkg_models():
                expectation += self._bkg_models[key]
        else:
            for key in self.keys_bkg_models():
                expectation += dict_bkg_norm.get(key, 1.0) * self._bkg_models[key]

        return np.where(expectation <= 0, almost_zero, expectation)

    def calc_T_product(self, dataspace_histogram):
        data_vec = np.asarray(dataspace_histogram, dtype=float)
        if data_vec.shape != (self._n_events,):
            raise ValueError(
                f"dataspace_histogram must have shape ({self._n_events},), "
                f"got {data_vec.shape}"
            )
        return self.response_matrix @ data_vec

    def calc_bkg_model_product(self, key, dataspace_histogram):
        data_vec = np.asarray(dataspace_histogram, dtype=float)
        return float(np.dot(self._bkg_models[key], data_vec))

    def calc_log_likelihood(self, expectation):
        expectation = np.where(np.asarray(expectation, dtype=float) <= 0, 1e-12, expectation)
        return float(np.sum(np.log(expectation) - expectation))



def unbinned_richardson_lucy(interface, model_init, n_iter=20, dict_bkg_norm=None):
    model = np.asarray(model_init, dtype=float).copy()
    log_likelihoods = []

    R_j = interface.exposure_map

    for _ in range(n_iter):
        expectation = interface.calc_expectation(model, dict_bkg_norm=dict_bkg_norm)
        log_likelihoods.append(interface.calc_log_likelihood(expectation))

        coeff = interface.calc_T_product(1.0 / expectation)

        norm_coeff = np.zeros_like(coeff)
        np.divide(coeff, R_j, out=norm_coeff, where=(R_j > 0))

        model *= norm_coeff

    return model, log_likelihoods

import matplotlib.pyplot as plt
import numpy as np
import healpy as hp

from cosipy.response.ideal_response import IdealComptonIRF,UnpolarizedIdealComptonIRF, RandomEventDataFromLineInSCFrame
from cosipy.polarization import StereographicConvention
# ============================================================
# Simulate events
# ============================================================
np.random.seed(42) # for reproducibility
def simulate_events():
    events = RandomEventDataFromLineInSCFrame(
        irf=UnpolarizedIdealComptonIRF.cosi_like(),
        flux=1. / (u.cm * u.cm * u.s),
        duration=1. * u.s,
        energy=1050 * u.keV,
        direction=SkyCoord(lon=0., lat=75., unit="deg", frame=SpacecraftFrame()),
        polarized_irf=IdealComptonIRF.cosi_like(),
        polarization_degree=0.2,
        polarization_angle=80. * u.deg,
        polarization_convention=StereographicConvention,
    )
    return events

events = simulate_events()

exposure_map = None 

interface = UnbinnedImageDataInterface(
    irf= UnpolarizedIdealComptonIRF.cosi_like(),
    events=events,
    nside=16,
    exposure_map=exposure_map,
    background_models={},
    radius_deg=None
)

model0 = np.ones(len(interface.pix_array), dtype=float) * 1e-6

model, log_like = unbinned_richardson_lucy(
    interface=interface,
    model_init=model0,
    n_iter=25,
    dict_bkg_norm={"background": 1.0},
)

# Full-sky HEALPix map
model_map = np.zeros(hp.nside2npix(16), dtype=float)
model_map[interface.pix_array] = model

hp.mollview(model_map, title="Deconvolved model", unit="arb", cmap="viridis")
plt.show()

plt.figure()
plt.plot(log_like)
plt.xlabel("Iteration")
plt.ylabel("Log likelihood")
plt.title("RL convergence")
plt.show()    