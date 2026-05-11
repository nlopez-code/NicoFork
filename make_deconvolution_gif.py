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
PKL_PATH    = "interface.pkl"
PARAMS_YAML = "deconvolution_params.yaml"
OUTPUT_DIR  = "deconvolution_frames"   # directory for per-iteration PNGs
GIF_PATH    = "deconvolution.gif"

FRAME_DURATION_MS = 600   # milliseconds per frame
LAST_FRAME_HOLD   = 3     # hold the final frame this many times longer

# Healpy projection settings
COORD        = "G"
CMAP         = "viridis"
GRATICULE    = True
LON_SPACING  = 30
LAT_SPACING  = 25
UNIT         = "arb"

# Marker for reference position (set to None to disable)
MARKER_LON_DEG = 0.0
MARKER_LAT_DEG = 0.0


def _render_frame(model_map, iteration, vmin, vmax):
    """Render one iteration to a matplotlib figure and return it."""
    hp.projview(
        model_map,
        title=f"Iteration {iteration}",
        unit=UNIT,
        cmap=CMAP,
        coord=COORD,
        graticule=GRATICULE,
        graticule_labels=GRATICULE,
        longitude_grid_spacing=LON_SPACING,
        latitude_grid_spacing=LAT_SPACING,
        min=vmin,
        max=vmax,
    )
    if MARKER_LON_DEG is not None:
        hp.newprojplot(
            MARKER_LON_DEG, MARKER_LAT_DEG,
            marker='o', lonlat=True, color='red', markersize=4,
        )
    return plt.gcf()


def _fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Run deconvolution ----
    print("Loading interface...")
    interface = UnbinnedImageDataInterface.load(PKL_PATH)
    dataset   = DataInterfaceCollection([interface])

    image_decon = ImageDeconvolution()
    image_decon.set_dataset(dataset)
    image_decon.read_parameterfile(PARAMS_YAML)
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

    # Use a consistent colour scale across all iterations
    vmin = 0.0
    vmax = float(np.max([m.max() for m in maps]))

    # ---- Render frames ----
    pil_frames = []
    for idx, (result, model_map) in enumerate(zip(results, maps)):
        iteration = result['iteration']
        print(f"  Rendering iteration {iteration} …")

        fig = _render_frame(model_map, iteration, vmin, vmax)

        # Save PNG
        png_path = os.path.join(OUTPUT_DIR, f"deconvolution_iter_{iteration:04d}.png")
        fig.savefig(png_path, dpi=150, bbox_inches='tight')

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
