import os
import re
import numpy as np
import matplotlib.pyplot as plt
import yaml

from astropy.time import Time
from astropy.table import Table, vstack
from histpy import Histogram

from cosipy.data_io.EmCDSUnbinnedData import TimeTagEmCDSEventDataInSCFrameFromArrays
from cosipy.response.ml import NFResponse, UnpolarizedNFFarFieldInstrumentResponseFunction
from cosipy.background_estimation.ml import FreeNormNFUnbinnedBackground, NFBackground
from cosipy.image_deconvolution.unbinned_spectral_data_interface import UnbinnedSpectralDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

# ============================================================
# Configuration
# ============================================================
FITS_PATHS  = [
    "dc4_mock_dataset_3months_unbinned_data_filtered_with_SAAcut_time_ordered.fits.gz",
    #"Positrons_Central_Source_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    #"Positrons_from_26Al_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    #"Positrons_from_44Ti_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    #"Broad_Bulge_511_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    #"Narrow_Bulge_511_3months_unbinned_data_filtered_with_SAAcut.fits.gz",
    #"positrons_thin_disk_line_3months_unbinned_data_filtered_with_SAAcut.fits.gz",

            ]

NF_RSP_PATH = "unpolarized_nfresponse_v1-00.pt"
LOAD_PKL    = True           # set True to skip rebuild and load cached interface

N_EVENTS    = None         # None = full dataset; int = evenly-spaced subsample

NSIDE       =  2            # sky-integration HEALPix resolution (NOT a model axis)

# Energy window and model binning are read from the YAML file.
with open("spectral_deconvolution_params.yaml") as _f:
    _params = yaml.safe_load(_f)
E_MIN_KEV = _params["energy_filter"]["min_keV"]
E_MAX_KEV = _params["energy_filter"]["max_keV"]
N_EBINS   = _params["energy_filter"]["n_bins"]

# Log-spaced model incident-energy bin edges spanning the energy window.
ENERGY_EDGES = np.geomspace(E_MIN_KEV, E_MAX_KEV, N_EBINS + 1)

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
_out_dir = os.path.join("Deconvolved Spectra", _subdir) if _subdir else "Deconvolved Spectra"
PKL_PATH = os.path.join(_pkl_dir, f"spec_ns{NSIDE}e{int(E_MIN_KEV)}{int(E_MAX_KEV)}nb{N_EBINS}n{_n_str}{_bkg}.pkl")

if __name__ == '__main__':
    # ============================================================
    # Load or build the interface
    # ============================================================
    if LOAD_PKL:
        interface = UnbinnedSpectralDataInterface.load(PKL_PATH)

        if BKG_PATH is not None:
            print("Building background model from binned estimate...")
            hist = Histogram.open(BKG_PATH)
            projected = hist.project(['Em', 'Phi', 'PsiChi'])
            bkg_rates = UnbinnedSpectralDataInterface.background_rates_from_binned_estimate(
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
            bkg_rates = UnbinnedSpectralDataInterface.background_rates_from_binned_estimate(
                projected, events
            )
            background_models['sky_model'] = bkg_rates
            print(f"Background model 'sky_model' ready ({len(bkg_rates)} per-event rates).")

        # --- Build interface and response matrix ---
        print("Building response matrix...")
        interface = UnbinnedSpectralDataInterface(
            irf               = irf,
            events            = events,
            energy_edges      = ENERGY_EDGES,
            sc_history        = sc_history,
            nside             = NSIDE,
            background_models = background_models or None,
        )
        _ = interface.response_matrix  # trigger build

        os.makedirs(_pkl_dir, exist_ok=True)
        interface.save(PKL_PATH)

    # ============================================================
    # Run deconvolution
    # ============================================================
    dataset      = DataInterfaceCollection([interface])
    spectr_decon = ImageDeconvolution()
    spectr_decon.set_dataset(dataset)
    spectr_decon.read_parameterfile("spectral_deconvolution_params.yaml")

    # Sync energy edges from the interface so the YAML doesn't need to be
    # updated manually when the binning changes.
    edges = interface.energy_edges
    spectr_decon.override_parameter(
        f"model_definition:property:energy_edges:value = {edges.value.tolist()}",
        f"model_definition:property:energy_edges:unit = {str(edges.unit)}",
    )

    spectr_decon.initialize()
    spectr_decon.run_deconvolution()

    final_model = spectr_decon.results[-1]['model']
    spectrum    = final_model.contents.value  # cm^-2 s^-1 keV^-1 sr^-1, sky-averaged
    e_edges     = final_model.axes['Ei'].edges.value
    e_mids      = np.sqrt(e_edges[:-1] * e_edges[1:])

    # Sky-integrated differential flux (uniform-emission model → × 4π sr)
    spectrum_sky = spectrum * 4 * np.pi

    has_bkg = BKG_NF_PATH is not None or BKG_PATH is not None
    bkg_str = " | bkg" if has_bkg else " | no bkg"
    e_str   = f"[{e_edges[0]:.0f}, {e_edges[-1]:.0f}] keV"
    title   = f"Sky Spectrum | E={e_str} | {N_EBINS} bins | N={interface._n_events:,}{bkg_str} | Iteration {len(spectr_decon.results)}"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stairs(spectrum_sky, e_edges, fill=False, linewidth=2)
    ax.plot(e_mids, spectrum_sky, 'o', markersize=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Incident energy [keV]')
    ax.set_ylabel(r'Sky-integrated flux [ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$]')
    ax.set_title(title, fontsize=10)
    ax.grid(True, which='both', alpha=0.3)

    if has_bkg:
        bkg_norms = spectr_decon.results[-1]['background_normalization']
        norm_str = ", ".join(f"{k}: {v:.3f}" for k, v in bkg_norms.items())
        ax.text(0.02, 0.02, f"bkg norm — {norm_str}", transform=ax.transAxes, fontsize=8)

    os.makedirs(_out_dir, exist_ok=True)
    fname = title.replace(" | ", "_").replace("[", "").replace("]", "").replace(",", "").replace(" ", "_")
    plt.savefig(os.path.join(_out_dir, fname + ".png"), dpi=150, bbox_inches="tight")

    plt.show()
