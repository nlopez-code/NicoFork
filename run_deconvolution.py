#!/usr/bin/env python3
"""Unbinned Richardson-Lucy deconvolution with the relative-coordinate IRF.

All settings are the constants below, the way the driver on origin/develop
does it.  Data handling follows that version too -- an explicit list of
FITS files, an energy cut, an optional evenly-spaced subsample, and the
full spacecraft history with no time window.

The response is this branch's IRFRelativeHistUnpolarized rather than
develop's NFResponse, which does not exist here.

``BACKGROUND`` selects the background component: ``"nf"`` for the trained
normalizing-flow model, ``"hist"`` for a histogrammed background
simulation, or ``None`` for no background at all.  See the setting itself
for what each one costs.

``INJECT_BACKGROUND`` sprinkles a reproducible random sample of *real*
background events from the DC4 background simulation into the event list,
so the background model has something to actually account for.  That is
the point of the exercise: inject a known number of background events and
see whether the fitted normalization recovers it.

Output goes to ``results/<key>/``, where the key is the one the response
matrix is cached under -- nside, energy band, event count, and the injected
background count and seed.  Two runs share a directory only when they share
a response matrix, so a run cannot leave files behind inside another's
results.

Writes iterations.json, model_iteration*.h5, exposure_map.h5, summary.json,
convergence.png, reconstructed_image.png, one mollview per iteration under
iterations/, and iterations_grid.png, plus a copy of this script and the
parameter file.  Exits non-zero if the log-likelihood ever decreased.

    python run_deconvolution.py
"""

import os
import re
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

import astropy.units as u
from astropy.time import Time

# ===========================================================================
# Configuration
# ===========================================================================
# expanduser: Python does not expand "~" the way the shell does, so a path
# left as "~/..." is a literal directory name that nothing matches.
data_dir = os.path.expanduser("~/software/testData")
FITS_PATHS =  [
    #data_dir + "/sources/dc4_mock_dataset_3months_unbinned_data_filtered_with_SAAcut_time_ordered.fits.gz",
    data_dir + "/sources/Positrons_Central_Source_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    data_dir + "/sources/Positrons_from_26Al_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    data_dir + "/sources/Positrons_from_44Ti_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    data_dir + "/sources/Broad_Bulge_511_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    data_dir + "/sources/Narrow_Bulge_511_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    data_dir + "/sources/positrons_thin_disk_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
            ]

IRF_PATH = data_dir + "/response/ResponseContinuum.area.relative.nonsparse.h5"
SC_PATH  = data_dir + "/orientation/DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"
PARFILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "imagedeconvolution_parfile.yml")

E_MIN_KEV = 505.0            # the model has one incident-energy bin, and the
E_MAX_KEV = 517.0            # IRF is evaluated at its midpoint -- keep it narrow

NSIDE     = 32               # 4 -> 192 pix, 8 -> 768 pix, 16 -> 3072 pix

N_EVENTS  = None             # None = every event that survives the energy cut

# --- Background --------------------------------------------------------
# One of:
#   None     no background component
#   "nf"     the trained normalizing-flow model (NFBackground).  Evaluates a
#            density for each *source* event, so nothing is added to the
#            event list and the response matrix does not grow.  This is the
#            cheap option, and the one the tutorial notebook uses.
#   "hist"   histogram a background *simulation* in (Em, Phi, PsiChi) and
#            interpolate it.  The simulation's events are merged into the
#            event list, so every one of them costs a response-matrix
#            column (npix * 8 bytes).  Needs histpy >= 2.0.7.
BACKGROUND = "nf"

BKG_LABEL = "mockset"        # key the deconvolution reports its norm under
BKG_FIT_NORM = True          # fit the norm instead of holding it at 1.0

# --- "nf" ---
# The flow is trained on the total DC4 background and carries its own
# absolute rate, so there is no background event file to read.
NF_BKG_PATH      = data_dir + "/background/nfbackground_v1-01.pt"
NF_BATCH_SIZE    = 100_000
# Must be explicit: FreeNormNFUnbinnedBackground opens the compute pool
# itself, and NFBase.init_compute_pool raises "no devices provided as
# argument or set as fallback" when this is None.  None here means "pick
# cuda if torch sees it, else cpu" -- see build_nf_background_model.
NF_DEVICES       = None
NF_COMPILE_MODE  = None      # torch.compile mode; None to skip compiling

# The flow is normalized over its whole training domain -- menergy_cuts =
# [100, 20000] keV in v1-01 -- so expected_counts() is the background over
# ALL of that range, while the events here are only the [E_MIN_KEV,
# E_MAX_KEV] slice.  N_tot has to be the integral over the slice we
# actually analyze, so it carries the fraction of the flow's counts that
# land in the band.  For 505-517 keV that fraction is ~0.0163, i.e. the
# uncorrected total is ~61x too large.
#
# None measures it by sampling the flow (a few minutes on CPU); set a float
# to skip the measurement, or 1.0 to reproduce the notebook's behaviour.
NF_BAND_FRACTION = None
NF_BAND_SAMPLES  = 20_000
NF_BAND_SEED     = 0

# --- "hist" ---
# The total DC4 background, all components including the SAA one.  It has
# to cover the same epoch as SC_PATH and FITS_PATHS.
#
# 168,648,544 events, and TimeTagEmCDSEventDataInSCFrameFromDC3Fits reads
# the whole table before the energy cut is applied -- see the note in the
# events section below.
BKG_PATHS = [
    data_dir + "/background/Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_withSAAbck.fits.gz",
            ]

# Binning of the (Em, Phi, PsiChi) distribution the per-event density is
# interpolated from.  This is a smoothing choice over the simulation, not a
# discretization of the model: with the total background there are enough
# events in the band to afford finer bins than the albedo-sized defaults
# these were picked for.  The run logs how many bins end up populated.
BKG_ENERGY_BINS  = 4
BKG_PHI_BINS     = 36
BKG_NSIDE        = 4         # PsiChi axis, in galactic coordinates

# --- Injected background events ----------------------------------------
# Sprinkle real background events from the DC4 background simulation into
# the source event list, so the background model has something to actually
# account for.  This is independent of BACKGROUND: the test is to inject a
# known number of background events and see whether the fitted background
# normalization recovers it.
#
# Every injected event costs a response-matrix column -- npix * 8 bytes,
# 96 kB at NSIDE=32 -- so this is the setting that decides whether the run
# still fits in RAM.  On top of the ~240k source events that already make a
# 22 GB matrix at this nside:
#
#    50,000 injected  ->  26.6 GB
#    75,000 injected  ->  28.9 GB
#   100,000 injected  ->  31.2 GB    over 32 GB of RAM once the IRF, the
#                                    event list and the flow are counted
#
# MAX_RESPONSE_GB below is the guard rail, and it is checked as soon as the
# event list is final -- before the IRF load, not after.
#
# The draw is reproducible: the same INJECT_N_EVENTS and INJECT_SEED select
# the same events out of the same file, every time.
INJECT_BACKGROUND = True
INJECT_N_EVENTS   = 75_000
INJECT_SEED       = 12345

# The DECOMPRESSED background file.  astropy cannot memmap a .gz, so given
# the .gz it decompresses all 168,648,544 rows x 112 bytes (18.9 GB) into
# memory before any cut is applied -- which does not fit alongside the
# response matrix.  Uncompressed, read_background_pool memmaps it and holds
# only INJECT_CHUNK_ROWS rows at a time.  Make it once with:
#
#   gunzip -k ~/software/testData/background/Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_withSAAbck.fits.gz
#
# -k keeps the .gz; the plain file needs 18.9 GB of disk.
INJECT_PATHS = [
    data_dir + "/background/Total_DC4_BG_3months_unbinned_data_filtered_with_SAAcut_withSAAbck.fits",
            ]

# Rows materialized per read.  Peak allocation is the five columns used,
# ~40 MB at a million rows, regardless of how big the file is.  (Resident
# memory looks larger while scanning because the memmapped pages count
# toward RSS; they are clean and the OS drops them under pressure.)
INJECT_CHUNK_ROWS = 1_000_000

CACHE     = True             # reuse the response matrix across runs

def run_key(nevents, n_background):
    """The identity of a run: what its response matrix is valid for.

    Everything in here changes the matrix, so a change to any of it has to
    produce a different name -- including the injection seed, since the same
    count with a different seed is a different set of events and therefore a
    different matrix.  Used for both the cache file and the results
    directory, so a result and the matrix that produced it carry the same
    name.
    """

    key = (f"ns{NSIDE}_e{int(E_MIN_KEV)}-{int(E_MAX_KEV)}_n{nevents}")
    if n_background:
        key += f"_bkg{n_background}"
        if INJECT_BACKGROUND:
            key += f"s{INJECT_SEED}"
    return key


# Refuse to build a response matrix bigger than this, rather than finding
# out by being OOM-killed an hour in.  It is npix * nevents * 8 bytes, and
# the deconvolution needs room for the IRF and the event list on top, so
# leave headroom below the machine's RAM.  None disables the check.
MAX_RESPONSE_GB = 29.0

# Cached response matrices go in the shared cache directory rather than
# OUT_DIR, so a results directory holds only the run's own output and stays
# comparable to one written by run_deconvolution.py.  They are ~1.5 GB each.
CACHE_DIR = os.path.expanduser("~/software/jar of pickles")

# Each run writes to <RESULTS_ROOT>/<key>, where the key is the same one
# the response matrix is cached under (see run_key).  Keying the directory
# means a run cannot land on top of a different one's output: two runs share
# a directory only when they share a response matrix, and then they really
# are the same run.  A fixed directory does not have that property -- a
# 50-iteration run dropped into one holding 125 files leaves the last 75
# behind, and nothing downstream can tell they are from something else.
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results")

SAVE_ALL_ITERATIONS = True   # False = only the last model_iteration*.h5
SAVE_PLOTS = True

# One mollview per iteration in <OUT_DIR>/iterations/, plus a contact sheet
# of all of them in iterations_grid.png.  Every frame shares one colour
# scale (see save_iteration_plots), so they can be compared directly.
SAVE_ITERATION_PLOTS = True
ITERATION_GRID_COLS  = 4     # columns in the contact sheet

# "shared"    one colour scale for every frame -- brightness growth is real
#             and comparable, but early iterations can look nearly blank.
# "per-frame" autoscale each frame -- structure is always visible, but the
#             frames are NOT comparable to each other.
ITERATION_PLOT_SCALE = "per-frame"

# Logarithmic colour scale for every sky map -- the final one, the
# per-iteration frames and the contact sheet.  A reconstructed 511 keV map
# spans decades between the bulge and the empty sky, which a linear scale
# renders as one bright blob on black.
PLOT_LOG_SCALE   = True
PLOT_LOG_DECADES = 6         # how far below the peak the colour scale runs

# cosipy may be pip-installed editable against a different checkout, which a
# bare `import cosipy` would silently pick up -- and IRFRelativeHistUnpolarized
# need not exist there.  Put this repo first.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
logging.getLogger("cosipy").setLevel(logging.INFO)
log = logging.getLogger("simple")


# ===========================================================================
# Event selection
# ===========================================================================

from cosipy.interfaces.event_selection import EventSelectorInterface


class EnergySelector(EventSelectorInterface):
    """Keep events in [emin, emax] keV.  cosipy ships no energy selector."""

    def __init__(self, emin_keV, emax_keV):
        self._emin = emin_keV
        self._emax = emax_keV

    def _select(self, events, early_stop=True):
        energy = np.asarray(events.energy_keV, dtype=float)
        return (energy >= self._emin) & (energy <= self._emax)


def _rebuild(events, idx):
    """A new event list holding ``events[idx]``, in that order."""

    from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays

    return TimeTagEmCDSEventDataInSCFrameFromArrays(
        jd1=np.asarray(events.jd1)[idx],
        jd2=np.asarray(events.jd2)[idx],
        energy_keV=np.asarray(events.energy_keV)[idx],
        scattered_lon_rad_sc=np.asarray(events.scattered_lon_rad_sc)[idx],
        scattered_lat_rad_sc=np.asarray(events.scattered_lat_rad_sc)[idx],
        scatt_angle_rad=np.asarray(events.scattering_angle_rad)[idx],
    )


def subsample(events, is_background, max_events):
    """Evenly-spaced subsample, (N_EVENTS in previous version).

    ``is_background`` is carried along because the background model's rate
    normalization is set from how many background events end up in the list
    the interface is actually given.
    """

    if max_events is None or events.nevents <= max_events:
        return events, is_background

    idx = np.linspace(0, events.nevents - 1, max_events, dtype=int)
    log.info("Subsampling %d -> %d events", events.nevents, len(idx))

    return _rebuild(events, idx), is_background[idx]


def concatenate(source_events, background_events):
    """Merge the source and background event lists into one time-sorted list.

    Returns the merged list and a boolean array flagging which of its events
    came from the background simulation.
    """

    if background_events is None or background_events.nevents == 0:
        return source_events, np.zeros(source_events.nevents, dtype=bool)

    lists = [source_events, background_events]

    def cat(attr):
        return np.concatenate([np.asarray(getattr(e, attr), dtype=float) for e in lists])

    jd1, jd2 = cat("jd1"), cat("jd2")
    is_background = np.concatenate([
        np.zeros(source_events.nevents, dtype=bool),
        np.ones(background_events.nevents, dtype=bool),
    ])

    order = np.argsort(jd1 + jd2)

    from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays

    merged = TimeTagEmCDSEventDataInSCFrameFromArrays(
        jd1=jd1[order],
        jd2=jd2[order],
        energy_keV=cat("energy_keV")[order],
        scattered_lon_rad_sc=cat("scattered_lon_rad_sc")[order],
        scattered_lat_rad_sc=cat("scattered_lat_rad_sc")[order],
        scatt_angle_rad=cat("scattering_angle_rad")[order],
    )

    return merged, is_background[order]


# ===========================================================================
# Background injection
# ===========================================================================
# The columns TimeTagEmCDSEventDataInSCFrameFromDC3Fits actually uses, under
# the names it gives them.  Everything else in the DC4 table -- the
# pointings, the distance, the galactic Chi/Psi -- is dead weight here.
_INJECT_COLUMNS = {
    "time_unix":  "TimeTags",
    "energy_keV": "Energies",
    "phi_rad":    "Phi",
    "chi_rad":    "Chi local",
    "psi_rad":    "Psi local",
}


def read_background_pool(paths, emin_keV, emax_keV, chunk_rows=INJECT_CHUNK_ROWS):
    """Every background event in [emin, emax] keV, read a chunk at a time.

    This exists instead of ``TimeTagEmCDSEventDataInSCFrameFromDC3Fits``
    because that reader (and astropy under it) materializes the whole table
    before the selection runs.  The total DC4 background is 168.6M rows of
    112 bytes, so that is 18.9 GB of event table before a single event has
    been cut -- and this run has to fit a 22 GB response matrix into the
    same 32 GB.

    Memmapping the *uncompressed* file and slicing rows keeps only
    ``chunk_rows`` rows resident.  The pages still stream past, so this
    reads the whole file from disk, but the resident cost is the chunk plus
    the surviving events.

    Returns a dict of native-endian float64 arrays keyed by
    ``_INJECT_COLUMNS``, holding the events inside the energy band.
    """

    from astropy.io import fits

    columns = {name: [] for name in _INJECT_COLUMNS}
    n_scanned = 0
    t0 = time.time()

    for path in paths:
        with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
            hdu = next((h for h in hdul if isinstance(h, fits.BinTableHDU)), None)
            if hdu is None:
                raise RuntimeError(f"{path}: no binary table extension")

            available = set(hdu.columns.names)
            missing = [c for c in _INJECT_COLUMNS.values() if c not in available]
            if missing:
                raise RuntimeError(f"{path}: missing column(s) {missing}")

            nrows = int(hdu.header["NAXIS2"])
            log.info("Scanning %s (%s rows) for events in [%g, %g] keV ...",
                     os.path.basename(path), f"{nrows:,}", emin_keV, emax_keV)

            # hdu.data on a memmapped file is a lazy record array; slicing it
            # is a view, and only the columns touched below are ever read.
            table = hdu.data

            for start in range(0, nrows, chunk_rows):
                rows = table[start:start + chunk_rows]

                # astype: FITS columns are big-endian, and torch refuses
                # anything but native order later on (see the sc_history
                # livetime note in the run section).
                energy = np.asarray(rows[_INJECT_COLUMNS["energy_keV"]]).astype(np.float64)
                keep = (energy >= emin_keV) & (energy <= emax_keV)

                if keep.any():
                    columns["energy_keV"].append(energy[keep])
                    for name, column in _INJECT_COLUMNS.items():
                        if name == "energy_keV":
                            continue
                        columns[name].append(
                            np.asarray(rows[column]).astype(np.float64)[keep])

                n_scanned += len(rows)
                if start and (start // chunk_rows) % 20 == 0:
                    rate = n_scanned / max(time.time() - t0, 1e-9)
                    log.info("... %5.1f%% (%s rows, %.1f M rows/s, ~%.0f s left)",
                             100.0 * n_scanned / nrows, f"{n_scanned:,}",
                             rate / 1e6, (nrows - n_scanned) / rate)

    pool = {name: (np.concatenate(parts) if parts else np.empty(0))
            for name, parts in columns.items()}

    n_in_band = pool["energy_keV"].size
    log.info("... %s of %s events in band (%.4g%%, %.1f s, %.0f MB held)",
             f"{n_in_band:,}", f"{n_scanned:,}",
             100.0 * n_in_band / max(n_scanned, 1), time.time() - t0,
             n_in_band * 8 * len(_INJECT_COLUMNS) / 1024**2)

    if n_in_band == 0:
        raise RuntimeError(
            f"No background event survived the [{emin_keV}, {emax_keV}] keV "
            "cut -- check INJECT_PATHS and the energy band.")

    return pool


def inject_background_events(pool, n_inject, seed, sc_history):
    """Draw ``n_inject`` background events at random from ``pool``.

    Returns the event list and the size of the pool it was drawn from: the
    number of *real* background events in the band and inside the
    spacecraft-history window.  That count is what the injected number has
    to be compared against, because their ratio is the fraction of the true
    background the analysis actually contains -- which is what the
    background model's prediction has to be scaled by.

    The window cut is not cosmetic.  ``FreeNormNFUnbinnedBackground``
    raises on any event outside ``[tstart, tstop]``, and an event landing
    exactly on ``tstop`` indexes one past the last livetime interval.
    """

    from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays

    tstart = float(sc_history.tstart.utc.unix)
    tstop = float(sc_history.tstop.utc.unix)

    inside = (pool["time_unix"] >= tstart) & (pool["time_unix"] < tstop)
    n_available = int(inside.sum())

    if n_available < pool["time_unix"].size:
        log.info("Dropping %s background event(s) outside the spacecraft "
                 "history window", f"{pool['time_unix'].size - n_available:,}")

    if n_available == 0:
        raise RuntimeError(
            "No in-band background event falls inside the spacecraft history "
            "window -- INJECT_PATHS and SC_PATH cover different epochs.")

    if n_inject > n_available:
        log.warning("Only %s in-band background events available; injecting "
                    "all of them instead of %s",
                    f"{n_available:,}", f"{n_inject:,}")
        n_inject = n_available

    # choice(replace=False) over the in-window subset, so the draw depends
    # only on the seed and the count -- not on where the window cut landed.
    rng = np.random.default_rng(seed)
    picked = np.flatnonzero(inside)[
        np.sort(rng.choice(n_available, size=n_inject, replace=False))]

    times = Time(pool["time_unix"][picked], format="unix")

    # Same column mapping as TimeTagEmCDSEventDataInSCFrameFromDC3Fits:
    # Chi local is the scattered longitude, Psi local its colatitude, and
    # Phi the Compton scattering angle.
    events = TimeTagEmCDSEventDataInSCFrameFromArrays(
        jd1=times.jd1,
        jd2=times.jd2,
        energy_keV=pool["energy_keV"][picked],
        scattered_lon_rad_sc=pool["chi_rad"][picked],
        scattered_lat_rad_sc=np.pi / 2 - pool["psi_rad"][picked],
        scatt_angle_rad=pool["phi_rad"][picked],
    )

    log.info("Injecting %s of %s in-band background events (seed %d) -- "
             "%.4g%% of the true background",
             f"{events.nevents:,}", f"{n_available:,}", seed,
             100.0 * events.nevents / n_available)

    return events, n_available


# ===========================================================================
# Background
# ===========================================================================

def check_background_prerequisites():
    """Fail before the expensive I/O if the background path cannot run.

    ``FreeNormBackground.__init__`` calls ``Histogram.to_dense(copy=...)``,
    which only exists in histpy >= 2.0.7 (what cosipy's pyproject.toml
    pins).  Older histpy fails deep inside the constructor with an opaque
    TypeError -- and by then the background event file has been read and the
    7 GB response loaded, which is a long wait for a bad message.
    """

    import histpy

    version = tuple(int(part) for part in histpy.__version__.split(".")[:3])
    if version < (2, 0, 7):
        raise RuntimeError(
            f"The background model needs histpy >= 2.0.7 (found "
            f"{histpy.__version__}): FreeNormBackground calls "
            "Histogram.to_dense(copy=...), which older versions do not "
            "accept.  Upgrade histpy, or set BACKGROUND = None.")


def build_background_model(simulation_events, events, n_background, sc_history):
    """A per-event background density from a background event simulation.

    ``simulation_events`` -- the background simulation on its own -- is
    histogrammed in ``(Em, Phi, PsiChi)``, PsiChi in galactic coordinates so
    the distribution is fixed on the sky rather than in the rotating
    spacecraft frame.  That gives the *shape*.  The full simulation is used,
    not the subsample, so the shape gets whatever statistics are available.

    ``events`` is the merged list the interface will be given: the model
    interpolates the distribution at each of *those* events, so its
    per-event densities line up with the interface's data axis.

    ``n_background`` is how many of ``events`` came from the simulation,
    counted after any subsampling.  It sets the *scale*: the rate is
    ``n_background / livetime``, so a deconvolution normalization of 1.0
    means "the counts the simulation predicts", which is the convention
    ``UnbinnedImageDataInterface`` expects.
    """

    from astropy.coordinates import SkyCoord
    from scoords import SpacecraftFrame
    from histpy import Axis, HealpixAxis, Histogram
    from cosipy.background_estimation.free_norm_threeml_binned_bkg import (
        FreeNormBackgroundInterpolatedDensityTimeTagEmCDS,
    )

    log.info("Building the '%s' background distribution from %d simulated "
             "events ...", BKG_LABEL, simulation_events.nevents)
    t0 = time.time()

    # --- The (Em, Phi, PsiChi) distribution ---
    energy_axis = Axis(
        np.linspace(E_MIN_KEV, E_MAX_KEV, BKG_ENERGY_BINS + 1) * u.keV,
        label="Em", scale="linear",
    )
    phi_axis = Axis(
        np.linspace(0.0, 180.0, BKG_PHI_BINS + 1) * u.deg,
        label="Phi", scale="linear",
    )
    psichi_axis = HealpixAxis(
        nside=BKG_NSIDE, scheme="ring", coordsys="galactic", label="PsiChi",
    )

    # Rotate each event's scattered direction into galactic coordinates,
    # the frame the PsiChi axis is defined in.
    times = Time(np.asarray(simulation_events.jd1),
                 np.asarray(simulation_events.jd2), format="jd")
    sc_dir = SkyCoord(np.asarray(simulation_events.scattered_lon_rad_sc),
                      np.asarray(simulation_events.scattered_lat_rad_sc),
                      unit=u.rad, frame=SpacecraftFrame())
    attitudes = sc_history.interp_attitude(times).transform_to("galactic")
    gal_vec = attitudes.rot.apply(sc_dir.cartesian.xyz.value.T)
    gal_dir = SkyCoord(gal_vec.T[0], gal_vec.T[1], gal_vec.T[2],
                       representation_type="cartesian", frame="galactic")

    distribution = Histogram([energy_axis, phi_axis, psichi_axis])
    distribution.fill(
        np.asarray(simulation_events.energy_keV) * u.keV,
        np.asarray(simulation_events.scattering_angle_rad) * u.rad,
        gal_dir,
    )

    filled = int(np.sum(distribution.contents > 0))
    log.info("... %d of %d (Em, Phi, PsiChi) bins populated",
             filled, distribution.contents.size)

    # `data` is the full event list, not the simulation: the model has to
    # produce one density per event the interface will see.
    model = FreeNormBackgroundInterpolatedDensityTimeTagEmCDS(
        data=events, distribution=distribution, sc_history=sc_history,
    )

    # Absolute scale: the rate implied by the background events actually in
    # the analysis list.
    livetime_s = sc_history.cumulative_livetime().to_value(u.s)
    rate_Hz = n_background / livetime_s
    model.set_norm(rate_Hz * u.Hz)

    log.info("... rate %.4g Hz -> %.1f expected counts over %.1f s "
             "(%.1f s wall)",
             rate_Hz, model.expected_counts(), livetime_s, time.time() - t0)

    return model


def build_nf_background_model(events, sc_history, thinning):
    """The trained normalizing-flow background, as a (density, total) pair.

    ``NFBackground`` carries its own absolute rate, so unlike the histogram
    path there is no simulation to read: the flow is evaluated at whatever
    events it is handed.

    ``thinning`` is the fraction of the true background the event list
    actually contains -- ``INJECT_N_EVENTS`` out of every in-band background
    event when injecting, or ``sampling_fraction`` when the list was thinned
    by ``N_EVENTS``.  Thinning a Poisson process by p gives a Poisson
    process of intensity p * lambda, so BOTH halves of the pair carry the
    factor: the total because it is the integral of that intensity, and the
    per-event density because it *is* that intensity evaluated at an event.
    Scaling only one of them leaves the pair mutually inconsistent, and no
    single fitted normalization can reconcile them -- the norm multiplies
    both together.

    The pair rather than the model object is deliberate:
    ``FreeNormNFUnbinnedBackground`` has no way to be told about the
    thinning, so registering it directly would hand the interface an
    unscaled total and bias both the fluxes and the fitted norm by 1 / p.

    The flow's total is also over its whole training domain, so it carries
    ``band_fraction`` on top -- see NF_BAND_FRACTION.  The density needs no
    such factor: it is already evaluated at in-band events.
    """

    from cosipy.background_estimation.ml import (
        FreeNormNFUnbinnedBackground,
        NFBackground,
    )

    import torch

    devices = NF_DEVICES
    if devices is None:
        devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]

    log.info("Loading the NF background from %s (devices=%s) ...",
             NF_BKG_PATH, devices)
    t0 = time.time()

    model = NFBackground(
        path_to_model=NF_BKG_PATH,
        density_batch_size=NF_BATCH_SIZE,
        devices=devices,
        density_compile_mode=NF_COMPILE_MODE,
        show_progress=False,
    )

    bkg = FreeNormNFUnbinnedBackground(
        model=model, data=events, sc_history=sc_history, label=BKG_LABEL,
    )

    log.info("Evaluating the flow at %d events ...", events.nevents)
    density = np.asarray(bkg.expectation_density(), dtype=float) * thinning
    full_total = float(bkg.expected_counts())

    band_fraction = NF_BAND_FRACTION
    if band_fraction is None:
        band_fraction = nf_band_fraction(model, sc_history)

    band_total = full_total * band_fraction
    total = band_total * thinning

    log.info("... %.4g counts over the flow's whole %s domain; x%.5g in "
             "band -> %.1f, x%.5g present -> N_tot = %.1f",
             full_total, "[100, 20000] keV", band_fraction, band_total,
             thinning, total)
    log.info("... per-event density min=%.4g max=%.4g (%.1f s total)",
             density.min(), density.max(), time.time() - t0)

    return density, total


def nf_band_fraction(model, sc_history):
    """Fraction of the flow's counts falling in [E_MIN_KEV, E_MAX_KEV].

    ``expected_counts()`` integrates the flow over its whole training
    domain, but the extended likelihood needs N_tot over the region of data
    space actually being analyzed -- here, one narrow energy slice.  That
    integral has no closed form, so estimate it by sampling: draw times from
    the counts-weighted history (livetime x rate, the distribution the
    background events themselves follow), sample the flow at them, and count
    what fraction lands in the band.
    """

    import torch

    log.info("Measuring the flow's [%g, %g] keV fraction from %d samples ...",
             E_MIN_KEV, E_MAX_KEV, NF_BAND_SAMPLES)
    t0 = time.time()

    obstime = sc_history.obstime
    mid = (obstime[:-1] + (obstime[1:] - obstime[:-1]) / 2).utc.unix
    mid = np.asarray(mid, dtype=np.float64)

    rate = np.asarray(
        model.evaluate_rate(torch.as_tensor(mid).view(-1, 1)), dtype=float
    ).ravel()
    livetime = np.asarray(sc_history.livetime.to_value(u.s), dtype=np.float64)

    weights = rate * livetime
    weights = weights / weights.sum()

    rng = np.random.default_rng(NF_BAND_SEED)
    idx = rng.choice(weights.size, size=NF_BAND_SAMPLES, p=weights)
    context = torch.as_tensor(mid[idx]).view(-1, 1)

    active = model.active_pool
    if not active:
        model.init_compute_pool()
    samples = model.sample_density(context)
    if not active:
        model.shutdown_compute_pool()

    # Column order matches FreeNormNFUnbinnedBackground._compute_density:
    # (energy_keV, phi_rad, scattered_lon, scattered colatitude).
    energy = np.asarray(samples[:, 0], dtype=float)
    fraction = float(np.mean((energy >= E_MIN_KEV) & (energy <= E_MAX_KEV)))

    if fraction == 0.0:
        raise RuntimeError(
            f"No sampled background event landed in [{E_MIN_KEV}, "
            f"{E_MAX_KEV}] keV -- raise NF_BAND_SAMPLES or set "
            "NF_BAND_FRACTION explicitly.")

    error = np.sqrt(fraction * (1.0 - fraction) / NF_BAND_SAMPLES)
    log.info("... fraction = %.5g +- %.2g (%.1f s).  Set NF_BAND_FRACTION "
             "to this to skip the measurement next time.",
             fraction, error, time.time() - t0)

    return fraction


# ===========================================================================
# Output
# ===========================================================================

def sky_norm(values, lo, hi):
    """Colour normalization for a sky map, logarithmic when asked for.

    Note this is a ``norm=`` object rather than any ``scale=`` argument.
    ``HealpixMap.plot`` forwards **kwargs straight to ``imshow``, which has
    no ``scale`` parameter -- passing one raises ``AxesImage.set() got an
    unexpected keyword argument``.  ``vmin``/``vmax`` must not be passed
    alongside a norm either; the limits live inside it.

    ``LogNorm`` renders anything <= 0 as blank, and a reconstructed map
    routinely has exact zeros in unexposed pixels, so the floor has to be
    positive.  A map whose smallest positive pixel sits many decades below
    the peak would also spend the whole colour range on numerical noise, so
    the floor is capped ``PLOT_LOG_DECADES`` below the maximum.
    """

    from matplotlib.colors import LogNorm, Normalize

    if not PLOT_LOG_SCALE:
        return Normalize(vmin=lo, vmax=hi)

    values = np.asarray(values)
    positive = values[values > 0]

    if hi <= 0 or positive.size == 0:
        log.warning("No positive pixels to plot; using a linear scale.")
        return Normalize(vmin=lo, vmax=hi)

    floor = max(float(positive.min()), float(hi) / 10.0 ** PLOT_LOG_DECADES)
    return LogNorm(vmin=floor, vmax=float(hi))


def clear_stale_iterations(out_dir, model_keep, frame_keep):
    """Delete per-iteration files this run is not going to write.

    A directory is one run's output, and a run that stops earlier than the
    one before it would otherwise leave that run's tail sitting there --
    ``model_iteration051.h5`` onwards from a 125-iteration run, in a
    directory whose summary.json says 50.  Nothing downstream can tell the
    difference, so an analysis reads the two as one sequence.

    Only the numbered per-iteration files are touched.  Everything else in
    the directory is either overwritten by this run (summary.json, the
    plots, the script copies) or was put there by hand.
    """

    stale = []

    for path in out_dir.glob("model_iteration*.h5"):
        match = re.fullmatch(r"model_iteration(\d+)\.h5", path.name)
        if match and int(match.group(1)) not in model_keep:
            stale.append(path)

    for path in (out_dir / "iterations").glob("iteration*.png"):
        match = re.fullmatch(r"iteration(\d+)\.png", path.name)
        if match and int(match.group(1)) not in frame_keep:
            stale.append(path)

    if not stale:
        return

    log.info("Removing %d stale per-iteration file(s) from an earlier run in "
             "%s (e.g. %s)", len(stale), out_dir, sorted(p.name for p in stale)[0])
    for path in stale:
        path.unlink()


def save_results(image_deconvolution, interface, out_dir, summary):
    """Write the reconstructed maps, the per-iteration table and diagnostics."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = image_deconvolution.results

    def expected_counts(model, dict_bkg_norm):
        expectation = interface.calc_expectation(model, dict_bkg_norm)
        return float(np.asarray(expectation.contents)[interface._i_norm])

    # --- Per-iteration table ---
    rows = []
    for r in results:
        model_map = np.asarray(r["model"].contents[:, 0].value)
        rows.append({
            "iteration": int(r["iteration"]),
            "log_likelihood": float(np.sum(r["log-likelihood"])),
            "expected_counts": expected_counts(model_map, r["background_normalization"]),
            "total_flux": float(np.sum(model_map)),
            "background_normalization": {
                k: float(v) for k, v in r["background_normalization"].items()
            },
        })

    with open(out_dir / "iterations.json", "w") as f:
        json.dump(rows, f, indent=2)

    # --- Model maps ---
    to_save = results if SAVE_ALL_ITERATIONS else results[-1:]

    # Before writing, not after: what is written now is what the directory
    # should hold, and clearing first means a crash between the two leaves a
    # short directory rather than a mixed one.
    clear_stale_iterations(
        out_dir,
        model_keep={r["iteration"] for r in to_save},
        frame_keep=({r["iteration"] for r in results}
                    if SAVE_PLOTS and SAVE_ITERATION_PLOTS else set()))

    for r in to_save:
        r["model"].write(str(out_dir / f"model_iteration{r['iteration']:03d}.h5"),
                         overwrite=True)

    interface.exposure_map.write(str(out_dir / "exposure_map.h5"), overwrite=True)

    # --- Run summary ---
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # No config file here -- the settings live in this script, so copy it
    # into the results directory alongside the parameter file, so a result
    # always records exactly what produced it.
    shutil.copy(os.path.abspath(__file__), out_dir / os.path.basename(__file__))
    shutil.copy(PARFILE, out_dir / "imagedeconvolution_parfile.yml")

    if not SAVE_PLOTS:
        return

    # --- Diagnostics ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mhealpy import HealpixMap
    except ImportError as e:
        log.warning("Skipping plots: %s", e)
        return

    iterations = [r["iteration"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(iterations, [r["log_likelihood"] for r in rows], marker=".")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("log-likelihood")
    ax1.grid(True)

    ax2.plot(iterations, [r["expected_counts"] for r in rows], marker=".",
             label=r"$N_\mathrm{tot}$")
    ax2.axhline(summary["n_events"], color="red", ls="--", label="observed events")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("expected counts")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=130)
    plt.close(fig)

    image = results[-1]["model"]
    final_values = np.asarray(image.contents[:, 0].value)
    healpix_map = HealpixMap(data=image[:, 0], unit=image.unit)
    fig, ax = healpix_map.plot(
        "mollview", cbar=True,
        norm=sky_norm(final_values, final_values.min(), final_values.max()))
    fig.colorbar.set_label(str(image.unit))
    ax.set_title(f"iteration {results[-1]['iteration']}, "
                 f"Ei = {image.axes['Ei'].bounds[0]}")
    fig.figure.savefig(out_dir / "reconstructed_image.png", dpi=130,
                       bbox_inches="tight")
    plt.close(fig.figure)

    if SAVE_ITERATION_PLOTS:
        save_iteration_plots(results, out_dir, plt, HealpixMap)


def save_iteration_plots(results, out_dir, plt, HealpixMap):
    """One mollview per iteration, plus a contact sheet of all of them.

    Every frame is drawn on the *same* colour scale.  That is the whole
    point: RL grows the dynamic range as it sharpens, so per-frame
    autoscaling makes each iteration look much like the last and hides
    exactly the convergence you are trying to watch.  The scale is taken
    from the brightest iteration, so early ones legitimately look faint.
    """

    frame_dir = out_dir / "iterations"
    frame_dir.mkdir(parents=True, exist_ok=True)

    maps = [np.asarray(r["model"].contents[:, 0].value) for r in results]
    unit = str(results[-1]["model"].unit)
    ei = results[-1]["model"].axes["Ei"].bounds[0]

    if ITERATION_PLOT_SCALE not in ("shared", "per-frame"):
        raise ValueError("ITERATION_PLOT_SCALE must be 'shared' or "
                         f"'per-frame', not {ITERATION_PLOT_SCALE!r}")

    shared = ITERATION_PLOT_SCALE == "shared"

    vmin = min(float(m.min()) for m in maps)
    vmax = max(float(m.max()) for m in maps)
    if vmax <= vmin:                       # a flat map would break imshow
        vmax = vmin + 1.0

    # One norm instance for every frame when the scale is shared -- that is
    # what makes the frames and the single colorbar mean the same thing.
    all_values = np.concatenate(maps)
    shared_norm = sky_norm(all_values, vmin, vmax) if shared else None

    def frame_norm(values):
        """Colour normalization for one frame -- the common one, or its own."""
        if shared:
            return shared_norm
        lo, hi = float(values.min()), float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        return sky_norm(values, lo, hi)

    log.info("Writing %d iteration images to %s (%s, %s colour scale, "
             "%.4g to %.4g %s)", len(maps), frame_dir, ITERATION_PLOT_SCALE,
             "log" if PLOT_LOG_SCALE else "linear", vmin, vmax, unit)

    # --- One file per iteration ---
    for r, values in zip(results, maps):
        healpix_map = HealpixMap(data=values, unit=r["model"].unit)
        img, ax = healpix_map.plot("mollview", cbar=True,
                                   norm=frame_norm(values))
        img.colorbar.set_label(unit)
        ax.set_title(f"iteration {r['iteration']}, Ei = {ei}")
        img.figure.savefig(frame_dir / f"iteration{r['iteration']:03d}.png",
                           dpi=130, bbox_inches="tight")
        plt.close(img.figure)

    # --- Contact sheet ---
    ncols = max(1, int(ITERATION_GRID_COLS))
    nrows = int(np.ceil(len(maps) / ncols))

    fig = plt.figure(figsize=(4.0 * ncols, 2.4 * nrows))
    last_img = None

    for i, (r, values) in enumerate(zip(results, maps), start=1):
        ax = fig.add_subplot(nrows, ncols, i, projection="mollview")
        healpix_map = HealpixMap(data=values, unit=r["model"].unit)
        last_img, ax = healpix_map.plot(ax=ax, cbar=False,
                                        norm=frame_norm(values))
        title = f"iter {r['iteration']}"
        if not shared:
            title += f"  ({values.max():.3g})"   # peak, since scales differ
        ax.set_title(title, fontsize=9)

    # Only meaningful when every panel shares it.
    if shared and last_img is not None:
        cbar = fig.colorbar(last_img, ax=fig.axes, fraction=0.02, pad=0.02)
        cbar.set_label(unit)

    if shared:
        lo_shown = shared_norm.vmin if shared_norm is not None else vmin
        fig.suptitle(f"Ei = {ei}, shared scale {lo_shown:.3g} to {vmax:.3g} "
                     f"{unit}", fontsize=10)
    else:
        fig.suptitle(f"Ei = {ei}, each panel autoscaled -- peak in {unit} "
                     "in parentheses", fontsize=10)
    fig.savefig(out_dir / "iterations_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Run
# ===========================================================================

if __name__ == "__main__":

    import cosipy
    from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromDC3Fits
    from cosipy.spacecraftfile import SpacecraftHistory
    from cosipy.response.relative_irf_hist import IRFRelativeHistUnpolarized
    from cosipy.interfaces.unbinned_image_data_interface import UnbinnedImageDataInterface
    from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
    from cosipy.image_deconvolution.data_interfaces.data_interface_collection import (
        DataInterfaceCollection,
    )

    # Only the parent here: the run's own directory is named after the event
    # list, which does not exist yet.  This still fails early if the path is
    # not writable at all.
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    t_run = time.time()

    # Fail here rather than a minute into the run, with the path that is
    # actually being looked for.
    if BACKGROUND not in (None, False, "nf", "hist"):
        raise SystemExit(f"BACKGROUND must be None, 'nf' or 'hist', "
                         f"not {BACKGROUND!r}")

    required = FITS_PATHS + [IRF_PATH, SC_PATH, PARFILE]
    if BACKGROUND == "nf":
        required = required + [NF_BKG_PATH]
    elif BACKGROUND == "hist":
        required = required + BKG_PATHS
    if INJECT_BACKGROUND:
        required = required + INJECT_PATHS
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        # A missing injection file is nearly always the .gz not having been
        # decompressed yet, so say so with the command rather than the path.
        hint = ""
        for path in missing:
            if path in INJECT_PATHS and os.path.exists(path + ".gz"):
                hint = (f"\n\nDecompress the background simulation first "
                        f"(needs 18.9 GB of disk):\n  gunzip -k {path}.gz")
        raise SystemExit("Missing input(s):\n  " + "\n  ".join(missing) + hint)

    # A .gz here would be read by astropy with no memmap: all 18.9 GB of the
    # table in RAM before the energy cut, which is exactly what the chunked
    # reader avoids.  Fail on it rather than let the machine swap to death.
    if INJECT_BACKGROUND:
        compressed = [p for p in INJECT_PATHS if p.endswith(".gz")]
        if compressed:
            raise SystemExit(
                "INJECT_PATHS must point at the DECOMPRESSED FITS file -- "
                "astropy cannot memmap a .gz and would read the whole "
                f"18.9 GB table into memory:\n  " + "\n  ".join(compressed) +
                "\n\nDecompress it once with:\n  gunzip -k " + compressed[0])

    # Likewise: say that histpy is too old now, not after the 7 GB response.
    if BACKGROUND == "hist":
        check_background_prerequisites()

    if INJECT_BACKGROUND and BACKGROUND is None:
        log.warning("Injecting background events with BACKGROUND = None: "
                    "nothing will model them, and the reconstruction will "
                    "absorb them into the sky.")

    # --- Events -----------------------------------------------------------
    # FromDC3Fits takes a list and time-sorts across files, so it already
    # does develop's vstack-and-argsort.
    log.info("Reading %d FITS file(s) ...", len(FITS_PATHS))
    source_events = TimeTagEmCDSEventDataInSCFrameFromDC3Fits(
        FITS_PATHS, selection=EnergySelector(E_MIN_KEV, E_MAX_KEV))
    log.info("... %d source events in [%g, %g] keV",
             source_events.nevents, E_MIN_KEV, E_MAX_KEV)

    # --- Spacecraft history ----------------------------------------------
    # Whole file, no time cut -- develop's behaviour.  The exposure map is
    # integrated over all of it, so this is only right when the events span
    # the same period; the two spans are logged so a mismatch is visible.
    #
    # Read before the background events, not after: injected events have to
    # be restricted to this window before they join the list, and the
    # fraction of the true background they represent is counted against the
    # events inside it.
    log.info("Reading the spacecraft history...")
    sc_history = SpacecraftHistory.open(SC_PATH)

    # astropy hands back FITS columns in the file's big-endian order, and
    # torch.as_tensor refuses anything but native order ("given numpy array
    # has byte order different from the native byte order").  That kills
    # FreeNormNFUnbinnedBackground._integrate_rate, which wraps
    # sc_history.livetime directly.  Normalize it once here.
    if np.asarray(sc_history.livetime).dtype.byteorder not in ("=", "|"):
        sc_history._livetime = u.Quantity(
            np.asarray(sc_history.livetime.to_value(u.s), dtype=np.float64), u.s)

    log.info("sc_hist %s to %s, livetime %.1f s",
             sc_history.tstart.utc.isot, sc_history.tstop.utc.isot,
             sc_history.livetime.sum().to_value(u.s))

    # --- Background events ------------------------------------------------
    # Background events ride along in the same list as the source events --
    # the likelihood does not distinguish the two, only the background
    # model's density does.
    bkg_events = None
    n_background_pool = 0

    if INJECT_BACKGROUND:
        pool = read_background_pool(INJECT_PATHS, E_MIN_KEV, E_MAX_KEV)
        bkg_events, n_background_pool = inject_background_events(
            pool, INJECT_N_EVENTS, INJECT_SEED, sc_history)
        del pool

        if BACKGROUND == "hist":
            log.warning("BACKGROUND = 'hist' with injection: the (Em, Phi, "
                        "PsiChi) shape is built from the %d injected events "
                        "alone, not the full simulation.", bkg_events.nevents)

    elif BACKGROUND == "hist":
        # This reader loads the entire table and only then applies the
        # selection (FromDC3Fits appends every column, time-sorts, and
        # passes `selection` to the parent), so peak memory here is set by
        # the file's total event count, not by how many survive the cut.
        # The total DC4 background is 168.6M events x 11 columns, and
        # astropy cannot memmap a .gz, so budget tens of GB.  Use
        # INJECT_BACKGROUND instead to stay within a sane memory budget.
        log.info("Reading %d background FITS file(s) ...", len(BKG_PATHS))
        bkg_events = TimeTagEmCDSEventDataInSCFrameFromDC3Fits(
            BKG_PATHS, selection=EnergySelector(E_MIN_KEV, E_MAX_KEV))
        log.info("... %d background events in [%g, %g] keV",
                 bkg_events.nevents, E_MIN_KEV, E_MAX_KEV)
        if bkg_events.nevents == 0:
            log.warning("No background events survived the energy cut; "
                        "running without a background model.")
            bkg_events = None

    events, is_background = concatenate(source_events, bkg_events)

    # Thinning a Poisson process with probability p gives a Poisson process
    # of intensity p * lambda, so every TOTAL downstream -- the exposure map
    # and the background's expected counts -- carries this factor.  Forget it
    # and every fitted flux is low by exactly p.
    n_in_band = events.nevents
    events, is_background = subsample(events, is_background, N_EVENTS)
    n_background = int(is_background.sum())
    sampling_fraction = events.nevents / n_in_band

    if sampling_fraction < 1.0:
        log.info("Sampling fraction %.5g -- the exposure map and the "
                 "background component are scaled by it", sampling_fraction)

    # The background component is thinned by its own factor, not the source
    # one: what fraction of the REAL background is in the list.  Injection
    # sets it explicitly (and N_EVENTS thins it further, which is why the
    # count is taken after the subsample); without injection the whole
    # background is nominally present and only N_EVENTS thins it.
    if INJECT_BACKGROUND:
        background_thinning = n_background / n_background_pool
        log.info("Background thinning %.5g (%s of %s in-band background "
                 "events) -- a correctly calibrated background model should "
                 "fit a normalization of 1.0",
                 background_thinning, f"{n_background:,}",
                 f"{n_background_pool:,}")
    else:
        background_thinning = sampling_fraction

    ev_times = Time(np.asarray(events.jd1), np.asarray(events.jd2), format="jd")
    log.info("events  %s to %s (%s events, %s from the background)",
             ev_times.min().utc.isot, ev_times.max().utc.isot,
             f"{events.nevents:,}", f"{n_background:,}")

    # --- Memory budget ----------------------------------------------------
    # Checked here, as soon as the event list is final, rather than at the
    # interface: everything between is expensive (a 7 GB IRF load, and the
    # flow evaluated at every event), and being told the list is too long
    # only after paying for that is no help at all.
    npix = 12 * NSIDE**2
    response_gb = npix * events.nevents * 8 / 1024**3
    log.info("Response matrix will be nside=%d (%d pixels) x %s events "
             "-> %.2f GB", NSIDE, npix, f"{events.nevents:,}", response_gb)

    if MAX_RESPONSE_GB is not None and response_gb > MAX_RESPONSE_GB:
        affordable = int(MAX_RESPONSE_GB * 1024**3 / (npix * 8))
        raise SystemExit(
            f"The response matrix would be {response_gb:.2f} GB, over the "
            f"{MAX_RESPONSE_GB:.2f} GB MAX_RESPONSE_GB budget.  At nside="
            f"{NSIDE} that budget buys {affordable:,} events and the list "
            f"holds {events.nevents:,}"
            + (f" -- {n_background:,} of them injected background, so drop "
               f"INJECT_N_EVENTS to {max(n_background - (events.nevents - affordable), 0):,} "
               f"or less" if n_background else "")
            + ".  Raise MAX_RESPONSE_GB to override.")

    # --- Identity of this run ---------------------------------------------
    # The event list is final, so the key is now known: it names both the
    # cached response matrix and the results directory, which is what ties a
    # result to the matrix that produced it.
    key = run_key(events.nevents, n_background)
    out_dir = os.path.join(RESULTS_ROOT, key)
    log.info("Run key %s -> %s", key, out_dir)

    # --- Response ---------------------------------------------------------
    # copy=False: the contents are ~7 GB and __init__ divides them by the
    # per-bin phase space in place.  Copying first would need a second 7 GB.
    log.info("Loading the IRF (~7 GB; needs roughly twice that while loading) ...")
    t0 = time.time()
    irf = IRFRelativeHistUnpolarized.from_h5(IRF_PATH, copy=False)
    log.info("... %.1f s", time.time() - t0)

    # --- Background model -------------------------------------------------
    # After the subsample, so the per-event densities line up with the event
    # list the interface is given.
    background_models = {}
    expected_background_norm = None

    if BACKGROUND == "nf":
        background_models[BKG_LABEL] = build_nf_background_model(
            events, sc_history, background_thinning)
    elif bkg_events is not None:
        background_models[BKG_LABEL] = build_background_model(
            bkg_events, events, n_background, sc_history)

    # With injection the answer is known: the list holds exactly
    # n_background background events, and the model was handed its
    # prediction for that same thinned process.  Their ratio is where the
    # fitted norm should land -- 1.0 if the model's absolute rate is right,
    # 1.3 if it underpredicts by 30%, and so on.
    if INJECT_BACKGROUND and background_models:
        predicted = float(background_models[BKG_LABEL][1]
                          if isinstance(background_models[BKG_LABEL], tuple)
                          else background_models[BKG_LABEL].expected_counts())
        expected_background_norm = n_background / predicted
        log.info("Injected %s background events against a predicted %.1f -- "
                 "the fitted '%s' norm should land near %.4f",
                 f"{n_background:,}", predicted, BKG_LABEL,
                 expected_background_norm)

    # --- Interface --------------------------------------------------------
    # The exposure map is integrated over the whole sc_history, so when the
    # events are a subsample of that span it has to carry the same factor:
    # it is the RL M-step denominator, and an unscaled one drags every flux
    # down by exactly sampling_fraction.  Take it from a probe interface --
    # the response matrix is lazy and never touched, so this costs only the
    # exposure integral the real interface would have run anyway.
    exposure_map = None
    if sampling_fraction < 1.0:
        probe = UnbinnedImageDataInterface(
            irf=irf,
            events=events,
            nside=NSIDE,
            sc_history=sc_history,
            energy_edges=[E_MIN_KEV, E_MAX_KEV],
        )
        contents = probe.exposure_map.contents
        if hasattr(contents, "value"):
            contents = contents.value
        exposure_map = np.asarray(contents, dtype=float)[:, 0] * sampling_fraction
        del probe

    interface = UnbinnedImageDataInterface(
        irf=irf,
        events=events,
        nside=NSIDE,
        sc_history=sc_history,
        energy_edges=[E_MIN_KEV, E_MAX_KEV],
        exposure_map=exposure_map,
        background_models=background_models or None,
    )

    # The response matrix is the whole cost of a run (~320k pixel-event
    # pairs/s); everything downstream is seconds.  Cache it on the settings
    # that determine it.
    cache_path = os.path.join(CACHE_DIR, f"response_{key}.npy")

    if CACHE:
        os.makedirs(CACHE_DIR, exist_ok=True)

    if CACHE and os.path.exists(cache_path):
        log.info("Loading the cached response matrix from %s", cache_path)
        interface._response_matrix = np.load(cache_path)
    else:
        t0 = time.time()
        matrix = interface.response_matrix
        log.info("Response matrix built in %.1f s: sum=%.4g", time.time() - t0, matrix.sum())
        if CACHE:
            np.save(cache_path, matrix)
            log.info("Cached the response matrix in %s", cache_path)

    if not np.any(interface.response_matrix > 0):
        raise RuntimeError(
            "The response matrix is identically zero -- usually a coordinate "
            "convention or energy range mismatch.")

    exposure = np.asarray(interface.exposure_map.contents)[:, 0]
    log.info("Exposure map: %.4g cm^2 s sr total, min=%.4g max=%.4g",
             exposure.sum(), exposure.min(), exposure.max())

    # --- Deconvolution ----------------------------------------------------
    image_decon = ImageDeconvolution()
    image_decon.set_dataset(DataInterfaceCollection([interface]))
    image_decon.read_parameterfile(PARFILE)

    # The interface's model axes are the authority, not the YAML.
    edges = interface.model_axes["Ei"].edges
    overrides = [
        f"model_definition:property:nside = {NSIDE}",
        f"model_definition:property:energy_edges:value = {edges.to('keV').value.tolist()}",
        "model_definition:property:energy_edges:unit = keV",
    ]

    if background_models:
        overrides.append(
            "deconvolution:parameter:background_normalization_optimization:activate = "
            f"{bool(BKG_FIT_NORM)}")

    image_decon.override_parameter(*overrides)

    image_decon.initialize()
    log.info("Running the deconvolution ...")
    t0 = time.time()
    image_decon.run_deconvolution()
    t_decon = time.time() - t0

    results = image_decon.results
    if not results:
        raise RuntimeError("The deconvolution produced no iterations.")

    log.info("... %d iteration(s) in %.1f s", len(results), t_decon)

    # --- Checks -----------------------------------------------------------
    log_likelihoods = np.array([float(np.sum(r["log-likelihood"])) for r in results])
    deltas = np.diff(log_likelihoods)
    monotonic = bool(np.all(deltas > -1e-6)) if deltas.size else True

    final_model = results[-1]["model"]
    final_map = np.asarray(final_model.contents[:, 0].value)
    n_tot = float(np.asarray(
        interface.calc_expectation(final_map,
                                   results[-1]["background_normalization"]).contents
    )[interface._i_norm])

    log.info("")
    log.info(f"{'iter':>5} {'<N_tot>':>15} {'log-likelihood':>18}")
    log.info("-" * 41)
    for r, ll in zip(results, log_likelihoods):
        model_map = np.asarray(r["model"].contents[:, 0].value)
        counts = float(np.asarray(
            interface.calc_expectation(model_map,
                                       r["background_normalization"]).contents
        )[interface._i_norm])
        log.info(f"{r['iteration']:>5} {counts:>15,.1f} {ll:>18,.2f}")
    log.info("")

    # RL cannot decrease the likelihood it maximizes, and MaxStepAccelerator
    # rejects any accelerated step that does.  A decrease here means the
    # likelihood itself is wrong -- most often the sc_history window not
    # matching the events.
    if monotonic:
        log.info("log-likelihood is monotonically non-decreasing: OK")
    else:
        log.warning("LOG-LIKELIHOOD DECREASED between iterations: %s",
                    deltas[deltas <= -1e-6])

    log.info("<N_tot> converged to %.1f against %d observed events "
             "(ratio %.3f)", n_tot, events.nevents, n_tot / events.nevents)
    log.info("final model: min=%.4g max=%.4g mean=%.4g %s",
             final_map.min(), final_map.max(), final_map.mean(),
             final_model.unit)

    # The point of an injection run: the number of background events in the
    # list is known exactly, so the fitted norm has a right answer.
    fitted_background_norm = results[-1]["background_normalization"].get(BKG_LABEL)
    if expected_background_norm is not None and fitted_background_norm is not None:
        fitted_background_norm = float(fitted_background_norm)
        log.info("'%s' norm fitted to %.4f against an expected %.4f "
                 "(ratio %.3f) for %s injected background events",
                 BKG_LABEL, fitted_background_norm, expected_background_norm,
                 fitted_background_norm / expected_background_norm,
                 f"{n_background:,}")

    # --- Output -----------------------------------------------------------
    summary = {
        "finished": datetime.now(timezone.utc).isoformat(),
        "config": os.path.abspath(__file__),
        "cosipy": os.path.dirname(cosipy.__file__),
        "irf": IRF_PATH,
        "irf_class": type(irf).__name__,
        "events_file": FITS_PATHS,
        "background_events_file": BKG_PATHS if BACKGROUND == "hist" else [],
        "nf_background_model": NF_BKG_PATH if BACKGROUND == "nf" else None,
        "orientation_file": SC_PATH,
        "tstart": sc_history.tstart.utc.isot,
        "tstop": sc_history.tstop.utc.isot,
        "livetime_s": float(sc_history.livetime.sum().to_value(u.s)),
        "energy_band_keV": [E_MIN_KEV, E_MAX_KEV],
        "nside": NSIDE,
        "npix": npix,
        "n_events": int(events.nevents),
        "n_source_events": int(events.nevents) - n_background,
        "n_background_events": n_background,
        "background_models": list(background_models),
        "background_type": BACKGROUND if background_models else None,
        "nf_band_fraction": NF_BAND_FRACTION,
        "background_fit_normalization": bool(BKG_FIT_NORM) if background_models else False,
        "sampling_fraction": sampling_fraction,
        # Everything needed to repeat an injection run exactly, and to judge
        # whether the background model accounted for what was injected.
        "inject_background": bool(INJECT_BACKGROUND),
        "inject_file": INJECT_PATHS if INJECT_BACKGROUND else [],
        "inject_requested": INJECT_N_EVENTS if INJECT_BACKGROUND else 0,
        "inject_seed": INJECT_SEED if INJECT_BACKGROUND else None,
        "background_pool_in_band": n_background_pool,
        "background_thinning": background_thinning,
        "expected_background_norm": expected_background_norm,
        "fitted_background_norm": (float(fitted_background_norm)
                                   if fitted_background_norm is not None else None),
        "n_iterations": len(results),
        "log_likelihood_first": float(log_likelihoods[0]),
        "log_likelihood_final": float(log_likelihoods[-1]),
        "log_likelihood_monotonic": monotonic,
        "expected_counts_final": n_tot,
        "total_flux_final": float(final_map.sum()),
        "deconvolution_seconds": t_decon,
        "total_seconds": time.time() - t_run,
    }

    save_results(image_decon, interface, out_dir, summary)
    log.info("Results written to %s", out_dir)

    raise SystemExit(0 if monotonic else 1)
