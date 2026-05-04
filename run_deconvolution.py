import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic
from astropy.time import Time
from astropy.table import Table
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

NSIDE        = 8 #Change in the YAML as well
RADIUS_DEG   = None
ENERGY       = 511 * u.keV
ENERGY_EDGES = u.Quantity([400., 600.], u.keV) #Change in the YAML as well

# Galactic source position (center of the pixel disc)
SOURCE_DIR = SkyCoord(l=0., b=0., unit="deg", frame=Galactic())

if __name__ == '__main__':
    # ============================================================
    # Load or build the interface
    # ============================================================
    if LOAD_PKL:
        interface = UnbinnedImageDataInterface.load(PKL_PATH)
    else:
        # --- Read FITS and time-sort everything together ---
        print("Reading FITS file...")
        t = Table.read(FITS_PATH)

        tsort = np.argsort(t['TimeTags'])
        t = t[tsort]

        # Energy filter to the line of interest
        mask = (t['Energies'] >= ENERGY_EDGES[0].value) & (t['Energies'] <= ENERGY_EDGES[1].value)
        t = t[mask]

        t = t[:5000]  # temporary limit for testing

        print(f"Events after energy cut: {len(t)}")

        # --- Build time-tagged events ---
        times = Time(t['TimeTags'].data, format='unix')

        events = TimeTagEmCDSEventDataInSCFrameFromArrays(
            jd1               = times.jd1,
            jd2               = times.jd2,
            energy_keV        = t['Energies'].data,
            scattered_lon_rad_sc = t['Chi local'].data,
            scattered_lat_rad_sc = np.pi / 2 - t['Psi local'].data,  # colatitude → latitude
            scatt_angle_rad   = t['Phi'].data,
        )

        # --- Build per-event Attitude from pointing columns ---
        # Xpointings/Zpointings shape: (N, 2), columns are [glon_rad, glat_rad]
        xp = t['Xpointings (glon,glat)'].data
        zp = t['Zpointings (glon,glat)'].data

        xpointings = SkyCoord(l=xp[:, 0], b=xp[:, 1], unit='rad', frame=Galactic())
        zpointings = SkyCoord(l=zp[:, 0], b=zp[:, 1], unit='rad', frame=Galactic())
        per_event_attitude = Attitude.from_axes(x=xpointings, z=zpointings, frame=Galactic())

        # --- Build IRF ---
        irf = UnpolarizedNFFarFieldInstrumentResponseFunction(
            NFResponse(NF_RSP_PATH, devices=["cpu"], area_compile_mode=None, density_compile_mode=None)
        )

        # --- Build interface ---
        print("Building response matrix...")
        interface = UnbinnedImageDataInterface(
            irf               = irf,
            events            = events,
            nside             = NSIDE,
            radius_deg        = RADIUS_DEG,
            background_models = {},
            source_dir        = SOURCE_DIR,
            energy            = ENERGY,
            energy_edges      = ENERGY_EDGES,
            attitude          = per_event_attitude,
        )

        _ = interface.response_matrix   # trigger build before saving
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
        title="Deconvolved model",
        unit="arb",
        cmap="viridis",
        graticule=True,
        graticule_labels=True,
        longitude_grid_spacing=30,
        latitude_grid_spacing=25,
        min=0.0,
        max=np.max(model_map),
    )
    hp.newprojplot(0.0, 0.0, marker='o', lonlat=True, color='red', markersize=3)
    plt.show()
