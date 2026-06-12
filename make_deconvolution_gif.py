"""
Run image deconvolution, save a PNG for every iteration, and write a GIF.

Outputs
-------
deconvolution_iter_NNNN.png  — one file per iteration (saved to OUTPUT_DIR)
deconvolution.gif            — animated GIF of all iterations
"""

import io
import os
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from PIL import Image

from cosipy.image_deconvolution.unbinned_image_data_interface import UnbinnedImageDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

# ============================================================
# Configuration
# ============================================================
COMPOSITE   = False              # set True when PKL came from a multi-file run
PKL_NAME    = "composites/ns16e505517nall.pkl"
PARAMS_YAML = "deconvolution_params.yaml"

_subdir    = "composites" if COMPOSITE else ""
_pkl_dir   = os.path.join("Jar of Pickles", _subdir) if _subdir else "Jar of Pickles"
PKL_PATH   = os.path.join(_pkl_dir, PKL_NAME)
OUTPUT_DIR = os.path.join("deconvolution_frames", _subdir) if _subdir else "deconvolution_frames"
GIF_PATH   = os.path.join(_subdir, "deconvolution.gif") if _subdir else "deconvolution.gif"

FRAME_DURATION_MS = 100   # milliseconds per frame
LAST_FRAME_HOLD   = 50     # hold the final frame this many times longer

# Healpy projection settings
COORD        = "G"
CMAP         = "plasma"
GRATICULE    = True
LON_SPACING  = 30
LAT_SPACING  = 30
UNIT         = r"cm$^{-2}$ s$^{-1}$ sr$^{-1}$"
LOG_SCALE    = True   # set False for linear colorbar

# Marker for reference position (set to None to disable)
MARKER_LON_DEG = None
MARKER_LAT_DEG = None

# Visual style
_BG_COLOR  = '#0D0F1A'   # deep cerulean background
_DPI       = 150          # 1920/150 = 12.8 in × 1080/150 = 7.2 in → 1920×1080 px
_FRAME_W   = 1920
_FRAME_H   = 930


def _render_frame(model_map, iteration):
    """Render one iteration to a matplotlib figure and return it."""
    rot = (MARKER_LON_DEG, MARKER_LAT_DEG) if MARKER_LON_DEG is not None else None
    if LOG_SCALE:
        vmin = max(float(model_map[model_map > 0].min()), float(model_map.max()) * 1e-4)
        norm = 'log'
    else:
        vmin = 0.0
        norm = None

    hp.projview(
        np.array(model_map, dtype=np.float64, copy=True),
        title=f"Iteration {iteration}",
        unit=UNIT,
        cmap=CMAP,
        coord=COORD,
        graticule=GRATICULE,
        graticule_labels=GRATICULE,
        longitude_grid_spacing=LON_SPACING,
        latitude_grid_spacing=LAT_SPACING,
        min=vmin,
        norm=norm,
        rot=rot,
        graticule_color='white',
    )

    fig = plt.gcf()
    fig.set_size_inches(_FRAME_W / _DPI, _FRAME_H / _DPI)
    fig.set_facecolor(_BG_COLOR)

    for ax in fig.get_axes():
        ax.set_facecolor(_BG_COLOR)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color('white')
        ax.title.set_color('white')

    # Catch any remaining black lines/text (graticule, colorbar ticks, etc.)
    for obj in fig.findobj(plt.Line2D):
        obj.set_color('white')
    for obj in fig.findobj(plt.Text):
        obj.set_color('white')

    if MARKER_LON_DEG is not None:
        if rot is not None:
            r = hp.Rotator(rot=list(rot), inv=False)
            colat_disp, lon_disp = r(
                np.deg2rad(90.0 - MARKER_LAT_DEG),
                np.deg2rad(MARKER_LON_DEG),
            )
            lon_plot = np.rad2deg(lon_disp)
            lat_plot = 90.0 - np.rad2deg(colat_disp)
        else:
            lon_plot = MARKER_LON_DEG
            lat_plot = MARKER_LAT_DEG
        hp.newprojplot(
            lon_plot, lat_plot,
            marker='o', lonlat=True, color='red', markersize=4,
        )

    return fig


def _fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL Image at exactly _FRAME_W × _FRAME_H."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=_DPI, facecolor=_BG_COLOR, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img.resize((_FRAME_W, _FRAME_H), Image.LANCZOS)


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if _subdir:
        os.makedirs(_subdir, exist_ok=True)  # parent dir for GIF_PATH

    # ---- Run deconvolution ----
    print("Loading interface...")
    interface = UnbinnedImageDataInterface.load(PKL_PATH)
    dataset   = DataInterfaceCollection([interface])

    image_decon = ImageDeconvolution()
    image_decon.set_dataset(dataset)
    image_decon.read_parameterfile(PARAMS_YAML)

    # Sync energy edges from the loaded interface so the YAML matches exactly.
    edges = interface.energy_edges
    image_decon.override_parameter(
        f"model_definition:property:energy_edges:value = {edges.value.tolist()}",
        f"model_definition:property:energy_edges:unit = {str(edges.unit)}",
    )

    image_decon.initialize()

    print("Running deconvolution...")
    image_decon.run_deconvolution()

    # ---- Collect maps ----
    results = image_decon.results
    maps = []
    for result in results:
        contents = result['model'].contents
        if hasattr(contents, 'value'):
            contents = contents.value
        maps.append(np.asarray(contents[:, 0], dtype=float))

    # ---- Render frames ----
    pil_frames = []
    for idx, (result, model_map) in enumerate(zip(results, maps)):
        iteration = result['iteration']
        print(f"  Rendering iteration {iteration} …")

        fig = _render_frame(model_map, iteration)

        # Save PNG
        png_path = os.path.join(OUTPUT_DIR, f"deconvolution_iter_{iteration:04d}.png")
        fig.savefig(png_path, dpi=_DPI, facecolor=_BG_COLOR, bbox_inches='tight')

        # Capture frame for GIF
        pil_frames.append(_fig_to_pil(fig))
        plt.close(fig)

    print(f"Saved {len(pil_frames)} PNGs to '{OUTPUT_DIR}/'")

    # ---- Write GIF ----
    if pil_frames:
        durations = [FRAME_DURATION_MS] * len(pil_frames)
        durations[-1] = FRAME_DURATION_MS * LAST_FRAME_HOLD  # linger on the final frame

        pil_frames[0].save(
            GIF_PATH,
            save_all=True,
            append_images=pil_frames[1:],
            duration=durations,
            loop=0,           # loop forever
            optimize=False,
        )
        print(f"GIF saved to '{GIF_PATH}'  ({len(pil_frames)} frames)")
