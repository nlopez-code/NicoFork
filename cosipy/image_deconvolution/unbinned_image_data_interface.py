import numpy as np
import healpy as hp
import mhealpy as mp
import logging
from astropy import units as u
from histpy import Histogram
from scoords.spacecraft_frame import SpacecraftFrame
from astropy.coordinates import SkyCoord
from cosipy.image_deconvolution.image_deconvolution_data_interface_base import ImageDeconvolutionDataInterfaceBase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class UnbinnedImageDataInterface(ImageDeconvolutionDataInterfaceBase):
    """
    Unbinned image data interface for RL-style deconvolution.

    Forward response is evaluated differentially in event phase space:
        p_i(j) = P(event_i | photon_j)

    The normalization in model space is stored separately in _exposure_map
    and should represent the full relevant phase space, not just the observed
    event list.
    """

    def __init__(
        self,
        irf,
        events,
        nside,
        exposure_map,
        background_models=None,
    ):
        super().__init__()

        self._irf = irf

        #For testing only
        if events is None:
            self._events_histogram = Histogram.read("events_histogram.h5")
            # Histogram contents and axes (For testing only)
            self._event_contents = self._event_histogram.contents
            self._energy_axis = self._event_histogram.axes["energy"]
            self._theta_axis = self._event_histogram.axes["theta"]
            self._phi_axis = self._event_histogram.axes["phi"]

        self._nside = nside
        self._npix = hp.nside2npix(self._nside)


        self._events = events

        
        self._n_events = len(self._events_histogram)

        self._event = 1.0

        self._exposure_map = np.asarray(exposure_map, dtype=float)
        if self._exposure_map.shape != (self._n_model,):
            raise ValueError(
                f"exposure_map must have shape ({self._n_model},), "
                f"got {self._exposure_map.shape}"
            )
        
        #Temp testing things
        source_dir = SkyCoord(lon=0., lat=75., unit="deg", frame=SpacecraftFrame())
        lon = source_dir.lon.to_value(u.rad)
        lat = source_dir.lat.to_value(u.rad)
        theta = 0.5 * np.pi - lat
        phi = lon                           
        vec = hp.ang2vec(theta, phi)
        radius_deg = 30
        radius_rad = np.deg2rad(radius_deg)
        pix_array = mp.query_disc(self._nside, vec, radius_rad)

        energy = 1050 * u.keV

        self._photons = []
        aeff = np.zeros(len(pix_array), dtype=float)
        for i, pix in enumerate(pix_array):
            theta_i, phi_i = hp.pix2ang(self._nside, pix)

            photon = self._irf.PhotonWithDirectionAndEnergyInSCFrame(
                    phi_i,
                    0.5 * np.pi - theta_i,
                    energy.to_value(u.keV),
                    )

            self._photons.append(photon)
            aeff[i] = self._irf.effective_area_cm2(photon)
        self._n_model = len(self._photons)
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

    def calc_expectation(self, events, model, dict_bkg_norm=None, almost_zero=1e-12):
        model = self._coerce_model(model)

        prob_matrix = np.zeros((len(model), events.nevents), dtype=float)

        for i, photon in enumerate(self._photons):
            prob_matrix[i, :] = np.array(list(self._irf.event_probability(photon, self._events_histogram)), dtype=float)

        if (i + 1) % 10 == 0 or (i + 1) == len(self._photons):
            logger.info("Computed %d / %d rows of probability matrix.", i + 1, len(self._photons))

            if dict_bkg_norm is None:
                for key in self.keys_bkg_models():
                    expectation += self._bkg_models[key]
            else:
                for key in self.keys_bkg_models():
                    expectation += dict_bkg_norm.get(key, 1.0) * self._bkg_models[key]

                return np.where(expectation <= 0, almost_zero, expectation)


    def calc_T_product(self, dataspace_histogram):
        data_vec = np.asarray(dataspace_histogram, dtype=float)
        return self._response_matrix @ data_vec

    def calc_bkg_model_product(self, key, dataspace_histogram):
        data_vec = np.asarray(dataspace_histogram, dtype=float)
        return float(np.dot(self._bkg_models[key], data_vec))

    def calc_log_likelihood(self, expectation):
        expectation = np.where(np.asarray(expectation, dtype=float) <= 0, 1e-12, expectation)
        return float(np.sum(np.log(expectation) - expectation))