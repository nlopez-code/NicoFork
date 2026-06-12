"""
Interactive viewer for image-deconvolution results.

A matplotlib slider lets you scrub through iterations in real time.
A Play / Pause button animates the sequence automatically.

Usage
-----
    python interactive_deconvolution.py
"""

import os
import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" or "MacOSX" if TkAgg isn't available
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

from cosipy.image_deconvolution.unbinned_image_data_interface import UnbinnedImageDataInterface
from cosipy.image_deconvolution.image_deconvolution import ImageDeconvolution
from cosipy.image_deconvolution.data_interfaces.data_interface_collection import DataInterfaceCollection

# ============================================================
# Configuration
# ============================================================
COMPOSITE   = False              # set True when PKL came from a multi-file run
PKL_NAME    = "ns8e505517n5e5.pkl"
PARAMS_YAML = "deconvolution_params.yaml"

_subdir  = "composites" if COMPOSITE else ""
_pkl_dir = os.path.join("Jar of Pickles", _subdir) if _subdir else "Jar of Pickles"
PKL_PATH = os.path.join(_pkl_dir, PKL_NAME)

FRAME_MS    = 600    # milliseconds per frame when playing

COORD       = "G"
CMAP        = "viridis"
GRATICULE   = True
LON_SPACING = 30
LAT_SPACING = 30
UNIT        = r"cm$^{-2}$ s$^{-1}$ sr$^{-1}$"
LOG_SCALE   = False   # set False for linear colorbar

MARKER_LON_DEG = None   # set to a float (e.g. 0.0) to draw a reference marker
MARKER_LAT_DEG = 0.0


# ============================================================
# Helpers
# ============================================================

def _render_to_array(model_map, iteration, dpi=120):
    """Render one healpy frame and return it as an (H, W, 3) uint8 array."""
    if LOG_SCALE:
        vmin = max(float(model_map[model_map > 0].min()), float(model_map.max()) * 1e-2)
        norm = 'log'
    else:
        vmin = 0.0
        norm = None
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
        norm=norm,
    )
    if MARKER_LON_DEG is not None:
        hp.newprojplot(
            MARKER_LON_DEG, MARKER_LAT_DEG,
            marker='o', lonlat=True, color='red', markersize=4,
        )
    fig = plt.gcf()
    fig.set_dpi(dpi)
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    # ---- Run deconvolution ----
    print("Loading interface …")
    interface = UnbinnedImageDataInterface.load(PKL_PATH)
    dataset   = DataInterfaceCollection([interface])

    image_decon = ImageDeconvolution()
    image_decon.set_dataset(dataset)
    image_decon.read_parameterfile(PARAMS_YAML)
    image_decon.initialize()

    print("Running deconvolution …")
    image_decon.run_deconvolution()

    results = image_decon.results
    n_iter  = len(results)
    print(f"Deconvolution finished — {n_iter} iteration(s).")

    # ---- Collect maps ----
    maps      = []
    iternums  = []
    for r in results:
        contents = r['model'].contents
        if hasattr(contents, 'value'):
            contents = contents.value
        maps.append(np.asarray(contents[:, 0], dtype=float))
        iternums.append(r['iteration'])

    # ---- Pre-render all frames ----
    print("Pre-rendering frames …")
    frames = []
    for i, (m, it) in enumerate(zip(maps, iternums)):
        print(f"  frame {i+1}/{n_iter}  (iteration {it})")
        frames.append(_render_to_array(m, it))
    print("Done.")

    # ---- Interactive figure ----
    fig_i, ax_i = plt.subplots(figsize=(11, 6))
    fig_i.subplots_adjust(bottom=0.18, top=0.98, left=0.01, right=0.99)
    ax_i.axis('off')

    img_display = ax_i.imshow(frames[0], aspect='auto')
    ax_i.set_title(f"Iteration {iternums[0]}", fontsize=12)

    # -- Slider --
    ax_slider = fig_i.add_axes([0.12, 0.08, 0.60, 0.035])
    slider    = Slider(
        ax_slider, 'Iteration',
        valmin=1, valmax=n_iter,
        valinit=1, valstep=1,
        color='steelblue',
    )
    # Show actual iteration number in the slider label
    slider.valtext.set_text(str(iternums[0]))

    def update(val):
        idx = int(slider.val) - 1
        img_display.set_data(frames[idx])
        ax_i.set_title(f"Iteration {iternums[idx]} - Galactic Coordinates", fontsize=12)
        slider.valtext.set_text(str(iternums[idx]))
        fig_i.canvas.draw_idle()

    slider.on_changed(update)

    # -- Play / Pause button --
    ax_play  = fig_i.add_axes([0.76, 0.07, 0.10, 0.05])
    btn_play = Button(ax_play, '▶  Play', color='0.85', hovercolor='0.75')

    _anim   = [None]
    _playing = [False]

    def _advance(_):
        next_val = int(slider.val) % n_iter + 1   # wrap at end
        slider.set_val(next_val)

    def toggle_play(event):
        if _playing[0]:
            _playing[0] = False
            btn_play.label.set_text('▶  Play')
            if _anim[0] is not None:
                _anim[0].event_source.stop()
        else:
            _playing[0] = True
            btn_play.label.set_text('⏸  Pause')
            _anim[0] = animation.FuncAnimation(
                fig_i, _advance, interval=FRAME_MS, cache_frame_data=False
            )
        fig_i.canvas.draw_idle()

    btn_play.on_clicked(toggle_play)

    # -- Reset button --
    ax_reset  = fig_i.add_axes([0.88, 0.07, 0.08, 0.05])
    btn_reset = Button(ax_reset, '⏮  Reset', color='0.85', hovercolor='0.75')

    def reset(event):
        if _playing[0]:
            toggle_play(None)
        slider.set_val(1)

    btn_reset.on_clicked(reset)

    plt.show()
