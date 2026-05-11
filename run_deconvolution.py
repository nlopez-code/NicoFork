import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic
from astropy.time import Time
from astropy.table import Table
from histpy import Histogram, Axes, Axis, HealpixAxis
from scipy.interpolate import RegularGridInterpolator
from scoords import Attitude

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml import NFResponse, UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.image_deconvolution.unbinned_image_data_interface import UnbinnedImageDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

# ============================================================
# Configuration
# ============================================================
FITS_PATH   = "positrons_thin_disk_cont_3months_unbinned_data_filtered_with_SAAcut.fits.gz"
NF_RSP_PATH = "unpolarized_nfresponse_v1-00.pt"
PKL_PATH    = "interface.pkl"
LOAD_PKL    = True          # set True to skip rebuild and load cached interface

N_EVENTS    = 50           # None = full dataset; int = evenly-spaced subsample

NSIDE       = 2              # sky model HEALPix resolution (change in YAML too)

# Background — set BKG_FITS_PATH to a GALPROP-format sky-model .dat file to enable.
BKG_FITS_PATH = "GalTotal_SA100_F98_input.dat"


def _parse_galprop_dat(path):
    """Parse a GALPROP-format sky-model .dat file.

    Returns
    -------
    lons_deg : np.ndarray (n_lon,)   galactic longitude bin centers [deg]
    lats_colat_deg : np.ndarray (n_lat,)  colatitude bin centers [deg]
                     (0 = north galactic pole, 180 = south)
    energies_keV : np.ndarray (n_energy,)  energy bin centers [keV]
    flux : np.ndarray (n_lon, n_lat, n_energy)
    """
    lons = lats = energies = None
    entries = []

    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == 'PA':
                lons = np.array(parts[1:], dtype=float)
            elif tag == 'TA':
                lats = np.array(parts[1:], dtype=float)
            elif tag == 'EA':
                energies = np.array(parts[1:], dtype=float)
            elif tag == 'AP':
                entries.append((int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])))

    flux = np.zeros((len(lons), len(lats), len(energies)), dtype=float)
    for l_idx, b_idx, e_idx, value in entries:
        flux[l_idx, b_idx, e_idx] = value

    return lons, lats, energies, flux


def _galprop_to_per_event_rates(lons_deg, lats_colat_deg, energies_keV,
                                flux, nside, event_energies_keV, response_sky):
    """Fold a GALPROP sky model through the response matrix → per-event rates.

    For each event i:
        B[i] = sum_j M(pixel_j, E_i) * R[j, i]

    The computation is batched over the model's energy bins to avoid a
    per-event interpolation loop.

    Parameters
    ----------
    lons_deg : np.ndarray (n_lon,)
    lats_colat_deg : np.ndarray (n_lat,)   colatitude [deg]
    energies_keV : np.ndarray (n_energy,)
    flux : np.ndarray (n_lon, n_lat, n_energy)
    nside : int    HEALPix nside of the sky model used in the interface.
    event_energies_keV : np.ndarray (n_events,)
    response_sky : np.ndarray (n_sky, n_events)   sky block of the response matrix.

    Returns
    -------
    np.ndarray, shape (n_events,)
    """
    npix = hp.nside2npix(nside)
    pix_theta, pix_phi = hp.pix2ang(nside, np.arange(npix))  # colatitude, lon [0, 2pi)
    pix_lon   = np.degrees(pix_phi)
    pix_colat = np.degrees(pix_theta)

    # Wrap longitudes into the PA range (≈ –180 to +180 deg)
    pix_lon = np.where(pix_lon > 180.0, pix_lon - 360.0, pix_lon)

    interp = RegularGridInterpolator(
        (lons_deg, lats_colat_deg, np.log10(energies_keV)),
        flux,
        method='linear',
        bounds_error=False,
        fill_value=0.0,
    )

    # Assign each event to its nearest model energy bin
    e_idx = np.clip(
        np.searchsorted(energies_keV, event_energies_keV, side='right') - 1,
        0, len(energies_keV) - 1,
    )

    rates = np.zeros(len(event_energies_keV), dtype=float)
    for k, e_k in enumerate(energies_keV):
        mask = e_idx == k
        if not mask.any():
            continue
        pts        = np.column_stack([pix_lon, pix_colat, np.full(npix, np.log10(e_k))])
        sky_flux_k = interp(pts)                        # (n_sky,)
        rates[mask] = response_sky[:, mask].T @ sky_flux_k

    return rates


if __name__ == '__main__':
    # ============================================================
    # Load or build the interface
    # ============================================================
    if LOAD_PKL:
        interface = UnbinnedImageDataInterface.load(PKL_PATH)
    else:
        # --- Read signal FITS and time-sort ---
        print("Reading signal FITS file...")
        t = Table.read(FITS_PATH)
        t = t[np.argsort(t['TimeTags'])]

        if N_EVENTS is not None:
            indices = np.linspace(0, len(t) - 1, min(N_EVENTS, len(t)), dtype=int)
            t = t[indices]

        print(f"Signal events: {len(t)}")

        # --- Build time-tagged events ---
        times = Time(t['TimeTags'].data, format='unix')
        events = TimeTagEmCDSEventDataInSCFrameFromArrays(
            jd1                  = times.jd1,
            jd2                  = times.jd2,
            energy_keV           = t['Energies'].data,
            scattered_lon_rad_sc = t['Chi local'].data,
            scattered_lat_rad_sc = np.pi / 2 - t['Psi local'].data,
            scatt_angle_rad      = t['Phi'].data,
        )

        # --- Per-event attitude from pointing columns ---
        xp = t['Xpointings (glon,glat)'].data
        zp = t['Zpointings (glon,glat)'].data
        xpointings = SkyCoord(l=xp[:, 0], b=xp[:, 1], unit='rad', frame=Galactic())
        zpointings = SkyCoord(l=zp[:, 0], b=zp[:, 1], unit='rad', frame=Galactic())
        per_event_attitude = Attitude.from_axes(x=xpointings, z=zpointings, frame=Galactic())

        # --- IRF ---
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(
            NFResponse(NF_RSP_PATH, devices=["cpu"],
                       area_compile_mode=None, density_compile_mode=None)
        )

        # --- Build interface and response matrix ---
        print("Building response matrix...")
        interface = UnbinnedImageDataInterface(
            irf      = irf,
            events   = events,
            nside    = NSIDE,
            attitude = per_event_attitude,
        )
        _ = interface.response_matrix  # trigger build

        # --- Background model from GALPROP .dat (optional) ---
        if BKG_FITS_PATH is not None:
            print("Building background model from GALPROP sky model...")
            lons, lats, energies, flux = _parse_galprop_dat(BKG_FITS_PATH)

            bkg_rates = _galprop_to_per_event_rates(
                lons, lats, energies, flux,
                nside              = NSIDE,
                event_energies_keV = np.asarray(events.energy_keV, dtype=float),
                response_sky       = interface.response_matrix[:interface._n_sky, :],
            )
            interface.add_background_model('gal_continuum', bkg_rates)
            print(f"Background model added ({len(bkg_rates)} per-event rates).")

        interface.save(PKL_PATH)

    # ============================================================
    # Run deconvolution
    # ============================================================
    dataset     = DataInterfaceCollection([interface])
    image_decon = ImageDeconvolution()
    image_decon.set_dataset(dataset)
    image_decon.read_parameterfile("deconvolution_params.yaml")
    image_decon.initialize()
    image_decon.run_deconvolution()

    final_model = image_decon.results[-1]['model']
    model_map   = (final_model.contents[:, 0]).value

    hp.projview(
        model_map,
        title="Galactic Coordinates",
        unit="arb",
        cmap="viridis",
        coord="G",
        graticule=True,
        graticule_labels=True,
        longitude_grid_spacing=30,
        latitude_grid_spacing=25,
        min=0.0,
        max=np.max(model_map),
    )

    plt.show()
