import pickle
import numpy as np
import healpy as hp
import mhealpy as mp
import logging
from astropy import units as u
from histpy import Histogram, Axes, Axis, HealpixAxis
from scoords.spacecraft_frame import SpacecraftFrame
from astropy.coordinates import SkyCoord
from scoords.attitude import Attitude
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame
from cosipy.image_deconvolution.data_interfaces.image_deconvolution_data_interface_base import (
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
        energy=None,
        energy_edges=None,
    ):
        super().__init__()

        self._irf = irf
        self._nside = nside
        self._npix = hp.nside2npix(self._nside)

        self._events = events

        self._n_events = events.nevents
        self._name = "UnbinnedImageDataInterface"
        self._exposure_map = None
        self._bkg_models = None
        self._summed_bkg_models = None
        self._energy = energy
        self._radius_deg = radius_deg

        # Define model-space pixels
        if source_dir is None:
            source_dir =  SkyCoord(lon=0., lat=75., unit="deg", frame=SpacecraftFrame())
        lon = source_dir.lon.to_value(u.rad)
        lat = source_dir.lat.to_value(u.rad)

        theta = 0.5 * np.pi - lat
        phi = lon

        vec = hp.ang2vec(theta, phi)
        
        if energy is None:
            energy = 1050 * u.keV

        if radius_deg == None:
            self._pix_array = np.arange(hp.nside2npix(self._nside))
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

        # Model axes: full-sky 2D (HealpixAxis + EnergyAxis) matching AllSkyImageModel
        if energy_edges is None:
            e_keV = energy.to_value(u.keV)
            energy_edges = u.Quantity([e_keV * 0.9, e_keV * 1.1], u.keV)
        image_axis  = HealpixAxis(nside=self._nside, scheme='ring', coordsys='galactic', label='lb')
        energy_axis = Axis(edges=energy_edges, label='Ei', scale='log')
        self._model_axes = Axes([image_axis, energy_axis])
        self._data_axes  = Axes([Axis(np.arange(self._n_events + 1), label='event')])

        # event histogram: one count per event
        self._event = Histogram(self._data_axes, contents=np.ones(self._n_events))

        if exposure_map is None:
            aeff_arr = aeff.copy()
        else:
            aeff_arr = np.asarray(exposure_map, dtype=float)
            if aeff_arr.shape != (self._n_model,):
                raise ValueError(
                    f"exposure_map must have shape ({self._n_model},), "
                    f"got {aeff_arr.shape}"
                )
        # embed per-pixel aeff into full-sky 2D Histogram
        aeff_fullsky = np.zeros((self._npix, 1), dtype=float)
        aeff_fullsky[self._pix_array, 0] = aeff_arr
        self._exposure_map = Histogram(self._model_axes, contents=aeff_fullsky)

        #if background_models is None:
        #    background_models = {"background": np.zeros(self._n_events, dtype=float)}

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
            self._summed_bkg_models[key] = float(np.sum(arr))

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
        if isinstance(model, Histogram):
            contents = model.contents
            if hasattr(contents, 'value'):
                contents = contents.value
            arr = np.asarray(contents[self._pix_array, 0], dtype=float).flatten()
        else:
            arr = np.asarray(model, dtype=float).flatten()
        if arr.shape != (self._n_model,):
            raise ValueError(
                f"model must have {self._n_model} elements, got {arr.shape}"
            )
        return arr

    def calc_source_expectation(self, model):
        arr = self._coerce_model(model)
        result = self.response_matrix.T @ arr
        return Histogram(self._data_axes, contents=result)

    def calc_bkg_expectation(self, dict_bkg_norm=None):
        expectation = np.zeros(self._n_events, dtype=float)
        if dict_bkg_norm is None:
            for key in self.keys_bkg_models():
                expectation += self._bkg_models[key]
        else:
            for key in self.keys_bkg_models():
                expectation += dict_bkg_norm.get(key, 1.0) * self._bkg_models[key]
        return Histogram(self._data_axes, contents=expectation)

    def calc_expectation(self, model, dict_bkg_norm=None, almost_zero=1e-12):
        combined = self.calc_source_expectation(model).contents + self.calc_bkg_expectation(dict_bkg_norm).contents
        combined = np.where(combined <= 0, almost_zero, combined)
        return Histogram(self._data_axes, contents=combined)

    def calc_T_product(self, dataspace_histogram):
        if isinstance(dataspace_histogram, Histogram):
            data_vec = dataspace_histogram.contents.flatten()
        else:
            data_vec = np.asarray(dataspace_histogram, dtype=float).flatten()
        result_partial = self.response_matrix @ data_vec  # shape (n_model,)
        result_fullsky = np.zeros((self._npix, 1), dtype=float)
        result_fullsky[self._pix_array, 0] = result_partial
        return Histogram(self._model_axes, contents=result_fullsky)

    def calc_bkg_model_product(self, key, dataspace_histogram):
        if isinstance(dataspace_histogram, Histogram):
            data_vec = dataspace_histogram.contents.flatten()
        else:
            data_vec = np.asarray(dataspace_histogram, dtype=float).flatten()
        return float(np.dot(self._bkg_models[key], data_vec))

    def calc_log_likelihood(self, expectation):
        if isinstance(expectation, Histogram):
            arr = expectation.contents.flatten()
        else:
            arr = np.asarray(expectation, dtype=float).flatten()
        arr = np.where(arr <= 0, 1e-12, arr)
        return float(np.sum(np.log(arr) - arr))

    def save(self, path):
        """Pickle the interface (including the cached response matrix) to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved UnbinnedImageDataInterface to %s", path)

    @classmethod
    def load(cls, path):
        """Load a previously saved interface from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Loaded UnbinnedImageDataInterface from %s", path)
        return obj

#Only for testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from cosipy.response.ideal_response import IdealComptonIRF, UnpolarizedIdealComptonIRF, RandomEventDataFromLineInSCFrame
    from cosipy.polarization import StereographicConvention
    from cosipy.image_deconvolution.algorithms.RichardsonLucyBasic import RichardsonLucyBasic
    from cosipy.image_deconvolution.models.allskyimage import AllSkyImageModel
    from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection


    # ============================================================
    # Simulate events
    # ============================================================
    np.random.seed(42)
    def simulate_events(nside):

        frame = SpacecraftFrame(attitude = Attitude.identity())
        coord = SkyCoord(lon=0., lat=75., unit="deg", frame=frame)

        # Nearest pixels
        m = HealpixAxis(nside=nside, scheme='ring', coordsys=frame, label='lb')

        #Centered at the center of the nearest pixel to the source direction
        coord = m.pix2skycoord(m.ang2pix(coord))

        return RandomEventDataFromLineInSCFrame(
            irf=UnpolarizedIdealComptonIRF.cosi_like(),
            flux=1. / (u.cm * u.cm * u.s),
            duration=1. * u.s,
            energy=1050 * u.keV,
            direction=coord,
            polarized_irf=IdealComptonIRF.cosi_like(),
            polarization_degree=0.2,
            polarization_angle=80. * u.deg,
            polarization_convention=StereographicConvention,
        )

    energy_edges = u.Quantity([945., 1155.], u.keV)  # single bin around 1050 keV
    nside = 64
    events = simulate_events(nside)

    interface = UnbinnedImageDataInterface(
        irf=UnpolarizedIdealComptonIRF.cosi_like(),
        events=events,
        nside=nside,
        radius_deg=45,
        background_models={},
        energy_edges=energy_edges,
    )

    _ = interface.response_matrix  # trigger build before saving
    interface.save("interface.pkl")

