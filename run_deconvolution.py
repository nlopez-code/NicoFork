import os
import re
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import yaml

from astropy.time import Time
from astropy.table import Table, vstack
from histpy import Histogram

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml import NFResponse, UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.background_estimation.ml import FreeNormNFUnbinnedBackground, NFBackground
from cosipy.image_deconvolution.unbinned_image_data_interface import UnbinnedImageDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

# ============================================================
# Configuration
# ============================================================
FITS_PATHS  = [
    "Positrons_Central_Source_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    "Positrons_from_26Al_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    "Positrons_from_44Ti_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    "Broad_Bulge_511_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    "Narrow_Bulge_511_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    "positrons_thin_disk_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",

            ]

NF_RSP_PATH = "unpolarized_nfresponse_v1-00.pt"
LOAD_PKL    = True           # set True to skip rebuild and load cached interface

N_EVENTS    = None           # None = full dataset; int = evenly-spaced subsample

NSIDE       =  16            # sky model HEALPix resolution (change in YAML too)

# Marker for reference position (set to None to disable)
MARKER_LON_DEG = None
MARKER_LAT_DEG = None

rot = (MARKER_LON_DEG, MARKER_LAT_DEG) if MARKER_LON_DEG is not None else None

# Energy window is read from the YAML file.
with open("deconvolution_params.yaml") as _f:
    _params = yaml.safe_load(_f)
E_MIN_KEV = _params["energy_filter"]["min_keV"]
E_MAX_KEV = _params["energy_filter"]["max_keV"]

# Background — provide exactly one of these:
#   BKG_NF_PATH  : path to an NFBackground model (.pt) + SC_FILE_PATH for SpacecraftHistory
#   BKG_PATH : path to a binned background estimate (.hdf5) from ContinuumEstimation
# Set the unused option to None.
BKG_NF_PATH  = 'nfbackground_v1-01.pt'
SC_FILE_PATH = "DC4_final_530km_3_month_with_slew_1sbins_GalacticEarth_SAA.fits"  # required
BKG_PATH = None

_n_str   = re.sub(r'e\+?0*(\d+)', r'e\1', f"{N_EVENTS:.0e}") if N_EVENTS is not None else "all"
_bkg     = "" if (BKG_NF_PATH is not None or BKG_PATH is not None) else "nb"
_subdir  = "composites" if len(FITS_PATHS) > 1 else ""
_pkl_dir = os.path.join("Jar of Pickles", _subdir) if _subdir else "Jar of Pickles"
_out_dir = os.path.join("Deconvolved Models", _subdir) if _subdir else "Deconvolved Models"
PKL_PATH = os.path.join(_pkl_dir, f"ns{NSIDE}e{int(E_MIN_KEV)}{int(E_MAX_KEV)}n{_n_str}{_bkg}.pkl")

if __name__ == '__main__':
    # ============================================================
    # Load or build the interface
    # ============================================================
    if LOAD_PKL:
        interface = UnbinnedImageDataInterface.load(PKL_PATH)

        if BKG_PATH is not None:
            print("Building background model from binned estimate...")
            hist = Histogram.open(BKG_PATH)
            projected = hist.project(['Em', 'Phi', 'PsiChi'])
            bkg_rates = UnbinnedImageDataInterface.background_rates_from_binned_estimate(
                projected, interface._events
            )
            interface.add_background_model('sky_model', bkg_rates)
            print(f"Background model 'sky_model' ready ({len(bkg_rates)} per-event rates).")

    else:
        # --- Read signal FITS and time-sort ---
        print(f"Reading {len(FITS_PATHS)} FITS file(s)...")
        t = vstack([Table.read(p) for p in FITS_PATHS], metadata_conflicts='silent')
        t = t[np.argsort(t['TimeTags'])]

        # Apply energy window first, then subsample so N_EVENTS counts post-cut events.
        energy_mask = (t['Energies'].data >= E_MIN_KEV) & (t['Energies'].data <= E_MAX_KEV)
        t = t[energy_mask]
        print(f"Signal events after energy cut [{E_MIN_KEV}, {E_MAX_KEV}] keV: {len(t)}")

        if N_EVENTS is not None:
            indices = np.linspace(0, len(t) - 1, min(N_EVENTS, len(t)), dtype=int)
            t = t[indices]
            print(f"Subsampled to {len(t)} events.")

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

        # --- Spacecraft history ---
        from cosipy.spacecraftfile import SpacecraftHistory
        sc_history = SpacecraftHistory.open(SC_FILE_PATH)

        # --- IRF ---
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(
            NFResponse(NF_RSP_PATH, devices=["cpu"],
                       area_compile_mode=None, density_compile_mode=None)
        )

        # --- Background models (resolved before interface construction) ---
        background_models = {}

        if BKG_NF_PATH is not None:
            print("Building NF background model...")
            nf_bkg = NFBackground(BKG_NF_PATH, devices=["cpu"], density_compile_mode=None)
            bkg = FreeNormNFUnbinnedBackground(nf_bkg, events, sc_history)
            background_models['nf_bkg'] = bkg
            print(f"NF background model ready ({events.nevents} per-event densities).")

        if BKG_PATH is not None:
            print("Building background model from binned estimate...")
            hist = Histogram.open(BKG_PATH)
            projected = hist.project(['Em', 'Phi', 'PsiChi'])
            bkg_rates = UnbinnedImageDataInterface.background_rates_from_binned_estimate(
                projected, events
            )
            background_models['sky_model'] = bkg_rates
            print(f"Background model 'sky_model' ready ({len(bkg_rates)} per-event rates).")

        # --- Build interface and response matrix ---
        print("Building response matrix...")
        interface = UnbinnedImageDataInterface(
            irf               = irf,
            events            = events,
            nside             = NSIDE,
            sc_history        = sc_history,
            energy_edges      = [505.0, 517.0],
            background_models = background_models or None,
        )
        _ = interface.response_matrix  # trigger build

        os.makedirs(_pkl_dir, exist_ok=True)
        interface.save(PKL_PATH)

    # ============================================================
    # Run deconvolution
    # ============================================================
    dataset     = DataInterfaceCollection([interface])
    image_decon = ImageDeconvolution()
    image_decon.set_dataset(dataset)
    image_decon.read_parameterfile("deconvolution_params.yaml")

    # Sync energy edges from the interface (min/max of event data) so the
    # YAML doesn't need to be updated manually when the dataset changes.
    edges = interface.energy_edges
    image_decon.override_parameter(
        f"model_definition:property:energy_edges:value = {edges.value.tolist()}",
        f"model_definition:property:energy_edges:unit = {str(edges.unit)}",
    )

    image_decon.initialize()
    image_decon.run_deconvolution()

    final_model = image_decon.results[-1]['model']
    model_map   = (final_model.contents[:, 0]).value

    has_bkg = BKG_NF_PATH is not None or BKG_PATH is not None
    bkg_str = " | bkg" if has_bkg else " | no bkg"
    e_edges = interface.energy_edges
    e_str   = f"[{e_edges.value[0]:.0f}, {e_edges.value[-1]:.0f}] {e_edges.unit}"
    title   = f"Galactic | nside={NSIDE} | E={e_str} | N={interface._n_events:,}{bkg_str} | Iteration {len(image_decon.results)}"

    hp.projview(
        model_map,
        title=title,
        unit=r"cm$^{-2}$ s$^{-1}$ sr$^{-1}$",
        cmap="viridis",
        coord="G",
        graticule=True,
        graticule_labels=True,
        longitude_grid_spacing=30,
        latitude_grid_spacing=30,
        rot=rot,
        norm='log',
        min=1e-8,
        max=np.max(model_map),
    )
 
    os.makedirs(_out_dir, exist_ok=True)
    fname = title.replace(" | ", "_").replace("[", "").replace("]", "").replace(",", "").replace(" ", "_")
    plt.savefig(os.path.join(_out_dir, fname + ".png"), dpi=150, bbox_inches="tight")

    plt.show()

