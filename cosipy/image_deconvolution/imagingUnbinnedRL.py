import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import mhealpy as mp
from histpy import Histogram, Axis

from scoords.spacecraft_frame import SpacecraftFrame
from astropy import units as u
from astropy.coordinates import SkyCoord

from cosipy.response.ideal_response import (
    IdealComptonIRF,
    UnpolarizedIdealComptonIRF,
    RandomEventDataFromLineInSCFrame,
)
from cosipy.response.photon_types import PhotonWithDirectionAndEnergyInSCFrame
from cosipy.polarization import StereographicConvention



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ===============================
# Configuration for events
# ===============================
energy = 1050 * u.keV

source_dir = SkyCoord(lon=0., lat=75., unit="deg", frame=SpacecraftFrame())
source_flux = 1. / (u.cm * u.cm * u.s)
duration = 1. * u.s
source_pd = 0.2
source_pa = 80. * u.deg
pol_convention = StereographicConvention()

#================================
# Healpix Configuration
#================================

order = 4
nside = 2 ** order
npix = hp.nside2npix(nside)

radius_deg = 30
radius_rad = np.deg2rad(radius_deg)

lon = source_dir.lon.to_value(u.rad)
lat = source_dir.lat.to_value(u.rad)

theta = 0.5 * np.pi - lat
phi = lon

vec = hp.ang2vec(theta, phi)

# Query pixels inside disc
pix_array = mp.query_disc(nside, vec, radius_rad)
logger.info("There are %d pixels in the disc.", len(pix_array))

# Blank map
hpx_map = np.full(npix, hp.UNSEEN)

# ===============================
# IRFs
# ===============================
logger.info("Loading IRFs...")
irf_pol = IdealComptonIRF.cosi_like()
irf_unpol = UnpolarizedIdealComptonIRF.cosi_like()

# ============================================================
# Simulate events
# ============================================================
np.random.seed(42) # for reproducibility
def simulate_events():
    events = RandomEventDataFromLineInSCFrame(
        irf=irf_unpol,
        flux=source_flux,
        duration=duration,
        energy=energy,
        direction=source_dir,
        polarized_irf=irf_pol,
        polarization_degree=source_pd,
        polarization_angle=source_pa,
        polarization_convention=pol_convention,
    )
    return events

events = simulate_events()

# event arrays from your simulation
energies = np.array(events.energy)
theta = np.array(events.scattered_lat_rad_sc)
phi = np.array(events.scattered_lon_rad_sc)

# define binning
energy_axis = Axis(np.linspace(0, 2000, 101), label="energy")
theta_axis = Axis(np.linspace(0, np.pi, 91), label="theta")
phi_axis = Axis(np.linspace(-np.pi, np.pi, 181), label="phi")

# create histogram
hist = Histogram([energy_axis, theta_axis, phi_axis])

# fill with events
hist.fill(energies, theta, phi)
hist.write("events_histogram.h5")
logger.info("Saved events to: event_histogram.h5")   
logger.info("Simulated %d events.", events.nevents)

# ============================================================
# Precompute photons and effective areas
# ============================================================
logger.info("Building photon list and effective areas...")
photons = []
aeff = np.zeros(len(pix_array), dtype=float)

for i, pix in enumerate(pix_array):
    theta_i, phi_i = hp.pix2ang(nside, pix)

    photon = PhotonWithDirectionAndEnergyInSCFrame(
        phi_i,
        0.5 * np.pi - theta_i,
        energy.to_value(u.keV),
    )

    photons.append(photon)
    aeff[i] = irf_unpol.effective_area_cm2(photon)

logger.info("Finished photon setup.")
# ============================================================
# Probability matrix
# ============================================================
logger.info("Calculating probability matrix...")
prob_matrix = np.zeros((len(pix_array), events.nevents), dtype=float)

for i, photon in enumerate(photons):
    prob_matrix[i, :] = np.array(list(irf_unpol.event_probability(photon, events)), dtype=float)

    if (i + 1) % 10 == 0 or (i + 1) == len(photons):
        logger.info("Computed %d / %d rows of probability matrix.", i + 1, len(photons))

logger.info("Probability matrix shape: %s", prob_matrix.shape)

# ============================================================
# Likelihood
# ============================================================
def poisson_binned_log_likelihood(observed, expected):
    expected_safe = np.where(expected <= 0, 1e-10, expected)
    return np.sum(observed * np.log(expected_safe) - expected_safe)

def unbinned_richardson_lucy(response, aeff, duration_s, model_init, n_iter=20, b_i=None):
    """
    Perform Richardson-Lucy deconvolution (Unbinned).

    Parameters:
        response (ndarray): Response matrix, shape (n_data, n_model)
        model_init (ndarray): Initial model vector, shape (n_model,)
        n_iter (int): Number of iterations

    Returns:
        model (ndarray): Deconvolved model
        log_likelihoods (list): Log-likelihood at each iteration
    """
    n_model, n_events = response.shape

    if b_i is None:
        b_i = np.zeros(n_events, dtype=float)
    model = model_init.copy()
    log_likelihoods = []

    R_j = aeff * duration_s

    for _ in range(n_iter):
        # expectation for each event
        expectation = response.T @ model + b_i
        expectation = np.where(expectation <= 0, 1e-10, expectation)

        log_likelihoods.append(poisson_binned_log_likelihood(1.0, expectation))

        # RL correction factor
        coeff = response @ (1.0 / expectation)

        norm_coeff = np.zeros_like(coeff)
        np.divide(coeff, R_j, out=norm_coeff, where=(R_j > 0))

        model *= norm_coeff

    return model, log_likelihoods

# ============================================================
# Initialization
# ============================================================
model = np.ones(prob_matrix.shape[0], dtype=float)
b_i = np.zeros(prob_matrix.shape[1], dtype=float)

logger.info("Starting Richardson-Lucy deconvolution...")

# ============================================================
# Output directory
# ============================================================
iterations_dir = Path(__file__).resolve().parent / "Iterations"
iterations_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Run RL iteratively and save each map
# ============================================================
n_iterations = 25
all_log_like = []

duration_s = duration.to_value(u.s)

# Run the deconvolution and save plots for each iteration
for i in range(n_iterations):
    logger.info("Running RL iteration %d / %d", i + 1, n_iterations)
    # Do ONE additional iteration each loop, instead of restarting from scratch
    model, log_like_step = unbinned_richardson_lucy(
        response=prob_matrix,
        aeff=aeff,
        duration_s=duration_s,
        model_init=model,
        n_iter=1,
        b_i=b_i,
    )
    all_log_like.extend(log_like_step)

    # Build full-sky plot map
    hpx_plot = np.full(npix, hp.UNSEEN, dtype=float)
    hpx_plot[pix_array] = model

    fig = plt.figure(figsize=(8, 5), dpi=150)

    hp.mollview(hpx_plot,
        fig=fig.number,
        title=f"Iteration {i+1}",
        unit="arb",
        cmap="viridis")

    hp.projplot(theta, phi,
            marker='o',
            color='red',
            markersize=1)

    outfile = iterations_dir / f"iteration_{i+1:03d}.png"
    plt.savefig(outfile,
                dpi=150,
                bbox_inches="tight")
    plt.close(fig)

logger.info("Done.")
logger.info("Saved iteration plots to %s", iterations_dir)
 