#!/usr/bin/env python3
"""Look at the model files a deconvolution run left behind.

``run_deconvolution.py`` writes one ``model_iteration*.h5`` per iteration
into its results directory.  Its own plots are made once, with whatever
colour scale the constants at the top of that script happened to say, and
regenerating them means rerunning the deconvolution.  This script reads the
saved models back instead, so the scale, the iteration and the output are
all decided after the fact.

Three subcommands:

    list    what is in a results directory, with the per-iteration range --
            run this first, the numbers are what you feed to --vmin/--vmax

    plot    one iteration as a mollview, with explicit limits -- written to
            a file and then opened in an image viewer, or with --window in an
            interactive one where vmin and vmax are sliders

    gif     every iteration stitched into an animation

Examples::

    python analyze_deconvolution.py list
    python analyze_deconvolution.py plot                       # last iteration
    python analyze_deconvolution.py plot -i 12
    python analyze_deconvolution.py plot -i 12 --vmin 1e-6 --vmax 1e-3
    python analyze_deconvolution.py plot -i -1 --linear -o final.png
    python analyze_deconvolution.py plot -i 12 --window     # vmin/vmax sliders
    python analyze_deconvolution.py plot -i 12 --no-show    # just the file
    python analyze_deconvolution.py plot --decades 3        # only the bright end

``--decades`` is the log floor when you do not set --vmin: it puts the
bottom of the colour scale that many decades below the peak.  A 511 keV map
peaking at 1e-2 gets a floor of 1e-5 at ``--decades 3``, which spends the
whole colour range on the bulge, or 1e-11 at ``--decades 9``, which brings
the faint halo out at the cost of contrast.  It also sets how far the
``--window`` sliders travel.
    python analyze_deconvolution.py gif
    python analyze_deconvolution.py gif --scale per-frame --fps 8
    python analyze_deconvolution.py gif --iterations 1-30:2 --vmax 6e-4

``run_deconvolution.py`` writes each run to ``results/<key>/``, named after
the response matrix it used -- nside, energy band, event count, injected
background count and seed.  With no directory argument these commands take
the most recently written run under ``results/``; pass one as the first
argument to pick another::

    python analyze_deconvolution.py gif results/ns32_e505-517_n313368_bkg75000s12345
"""

import os
import re
import sys
import glob
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np

# cosipy may be pip-installed editable against a different checkout, which a
# bare ``import cosipy`` would silently pick up.  Put this repo first, the
# same way run_deconvolution.py does.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RESULTS_ROOT = os.path.join(REPO_ROOT, "results")


def default_results_dir():
    """The most recently written run under ``results/``.

    ``run_deconvolution.py`` names each run's directory after its response
    key, so there is no fixed path to default to any more.  Ranked by the
    newest model file rather than the directory's own mtime, which merely
    reordering the files would disturb.
    """

    runs = []
    for entry in glob.glob(os.path.join(RESULTS_ROOT, "*")):
        models = glob.glob(os.path.join(entry, "model_iteration*.h5"))
        if models:
            runs.append((max(os.path.getmtime(m) for m in models), entry))

    if not runs:
        return RESULTS_ROOT          # nothing to find; let find_models say so

    return max(runs)[1]

# Frames of an animated GIF must all be the same size, so the figure size is
# fixed here rather than trimmed per frame with bbox_inches="tight" (which
# gives a slightly different size whenever a tick label changes width).
FIG_SIZE = (7.0, 4.2)
FIG_DPI  = 130


# ===========================================================================
# Reading a results directory
# ===========================================================================

def find_models(results_dir):
    """Return [(iteration, path), ...] sorted by iteration."""

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise SystemExit(f"No such results directory: {results_dir}")

    found = []
    for path in glob.glob(str(results_dir / "model_iteration*.h5")):
        match = re.search(r"model_iteration(\d+)\.h5$", path)
        if match:
            found.append((int(match.group(1)), Path(path)))

    if not found:
        raise SystemExit(f"No model_iteration*.h5 files in {results_dir}")

    return sorted(found)


def load_model(path):
    """Open one saved model.  Imported lazily -- cosipy takes seconds."""

    from cosipy.image_deconvolution.models.allskyimage import AllSkyImageModel

    return AllSkyImageModel.open(str(path))


def model_values(model, energy_index):
    """The sky map for one energy bin, as a plain float array."""

    n_energy = model.contents.shape[1]
    if not -n_energy <= energy_index < n_energy:
        raise SystemExit(f"--energy-index {energy_index} out of range; the "
                         f"model has {n_energy} energy bin(s)")

    return np.asarray(model.contents[:, energy_index].value)


def parse_iterations(spec, available):
    """Turn ``"1-30:2"`` / ``"1,5,9"`` / ``"12"`` into a list of iterations.

    ``available`` is the sorted list of iterations actually on disk; anything
    selected that is not there is an error rather than a silent gap, since a
    missing iteration in an animation is exactly the kind of thing you would
    not notice and would then misread.
    """

    if spec is None:
        return list(available)

    wanted = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)-(-?\d+)(?::(\d+))?", part)
        if match:
            lo, hi = int(match.group(1)), int(match.group(2))
            step = int(match.group(3) or 1)
            if step < 1:
                raise SystemExit(f"Step must be >= 1 in {part!r}")
            wanted.extend(range(lo, hi + 1, step))
        elif re.fullmatch(r"-?\d+", part):
            wanted.append(int(part))
        else:
            raise SystemExit(f"Cannot parse --iterations {part!r}; expected "
                             "N, A-B, A-B:step, or a comma-separated list")

    missing = sorted(set(wanted) - set(available))
    if missing:
        shown = ", ".join(str(m) for m in missing[:8])
        if len(missing) > 8:
            shown += f", ... ({len(missing)} in total)"
        raise SystemExit(f"No saved model for iteration(s): {shown}; have "
                         f"{available[0]}-{available[-1]}")

    # De-duplicate but keep the order asked for -- a reversed or shuffled
    # selection is a legitimate thing to want out of an animation.
    seen, ordered = set(), []
    for i in wanted:
        if i not in seen:
            seen.add(i)
            ordered.append(i)
    return ordered


def resolve_iteration(requested, available):
    """One iteration, where negatives count back from the last one."""

    if requested is None:
        return available[-1]
    if requested < 0:
        if -requested > len(available):
            raise SystemExit(f"Only {len(available)} iterations saved")
        return available[requested]
    if requested not in available:
        raise SystemExit(f"No saved model for iteration {requested}; have "
                         f"{available[0]}-{available[-1]}")
    return requested


# ===========================================================================
# Colour scale
# ===========================================================================

def sky_norm(source, vmin, vmax, log, decades):
    """Colour normalization for a sky map, logarithmic when asked for.

    ``source`` is the array the limits are taken from when --vmin/--vmax are
    not given: the frame itself, or every frame at once for a shared scale.
    An explicit limit always wins over anything derived here.

    This is a ``norm=`` object rather than any ``scale=`` argument.
    ``HealpixMap.plot`` forwards **kwargs straight to ``imshow``, which has
    no ``scale`` parameter, and ``vmin``/``vmax`` must not be passed
    alongside a norm either -- the limits live inside it.

    ``LogNorm`` renders anything <= 0 as blank, and a reconstructed map
    routinely has exact zeros in unexposed pixels, so a log floor has to be
    positive.  It also has to be capped ``decades`` below the top: RL drives
    empty pixels down to ~1e-40 while the bulge sits at ~1e-2, and a colour
    bar spanning all 38 decades is a colour bar of numerical noise.
    """

    from matplotlib.colors import LogNorm, Normalize

    source = np.asarray(source)
    top = float(vmax) if vmax is not None else float(source.max())

    if not log:
        bottom = float(vmin) if vmin is not None else float(source.min())
        if top <= bottom:                 # a flat map would break imshow
            top = bottom + 1.0
        return Normalize(vmin=bottom, vmax=top)

    positive = source[source > 0]

    if top <= 0 or positive.size == 0:
        print("No positive pixels to plot; falling back to a linear scale.")
        bottom = float(vmin) if vmin is not None else float(source.min())
        return Normalize(vmin=bottom, vmax=max(top, bottom + 1.0))

    if vmin is not None and vmin > 0:
        bottom = float(vmin)
    else:
        if vmin is not None:
            print(f"--vmin {vmin:g} is not positive and a log scale cannot "
                  "show it; using the smallest positive pixel instead.")
        bottom = max(float(positive.min()), top / 10.0 ** decades)

    if top <= bottom:
        top = bottom * 10.0

    return LogNorm(vmin=bottom, vmax=top)


def norm_for(args, all_values, frame_values):
    """The normalization for one frame under the chosen --scale.

    "shared" derives the limits from every selected iteration at once, so
    brightness growth across the animation is real and comparable;
    "per-frame" derives them from each frame, so structure is always visible
    but the frames do NOT mean the same thing.  --vmin/--vmax override
    either way.
    """

    source = all_values if args.scale == "shared" else frame_values
    return sky_norm(source, args.vmin, args.vmax, args.log, args.decades)


# ===========================================================================
# Drawing
# ===========================================================================

def make_figure(values, model, unit, title, norm, cmap, cbar=True, ax=None):
    """One mollview.  Returns (figure, image) -- the image is the mappable,
    which is what an interactive caller needs to restyle in place."""

    import matplotlib.pyplot as plt
    from mhealpy import HealpixMap

    if ax is None:
        fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        ax = fig.add_subplot(111, projection="mollview")
    else:
        fig = ax.figure

    healpix_map = HealpixMap(data=values, unit=model.unit)
    img, ax = healpix_map.plot(ax=ax, cbar=cbar, norm=norm, cmap=cmap)

    if cbar and getattr(img, "colorbar", None) is not None:
        img.colorbar.set_label(unit)
    ax.set_title(title)

    return fig, img


def use_agg(window):
    """Pick a matplotlib backend.  Must run before pyplot is imported."""

    import matplotlib

    if not window:
        matplotlib.use("Agg")


def open_in_viewer(path):
    """Show a written image in whatever the desktop uses for images.

    The point is that the file is always written first, so this is only
    ever a convenience: if there is no viewer -- over ssh, in a container --
    the plot is still on disk and the failure is worth one line, not a
    traceback.
    """

    import subprocess

    if sys.platform == "darwin":
        command = ["open", str(path)]
    elif sys.platform.startswith("win"):
        os.startfile(str(path))            # noqa: S606 -- Windows has no opener
        return True
    else:
        command = ["xdg-open", str(path)]

    try:
        subprocess.Popen(command,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as e:
        print(f"Could not open a viewer ({e}); the image is at {path}")
        return False
    return True


def frame_title(iteration, model, args, note=""):
    ei = model.axes["Ei"].bounds[args.energy_index]
    title = f"iteration {iteration}, Ei = {ei}"
    if note:
        title += f"  {note}"
    return title



def interactive_window(values, model, unit, title, norm, args, out):
    """Show the map with vmin/vmax sliders, and return the limits chosen.

    The sliders move in log10 when the scale is logarithmic: a linear slider
    over a range spanning six decades would spend nearly all its travel in
    the top decade and be useless for setting a floor.  Each starts near the
    place on its own track: vmin near the top, with nearly all the travel
    below it, and vmax in the middle, with room either way.

    Nothing is recomputed as they move -- the pixel values do not change, only
    the normalization does -- so this stays responsive even at nside 32 and
    up.  The limits are printed on exit as ready-to-paste flags, which is how
    a scale tuned here gets into ``gif`` or a scripted rerun.
    """

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    log = args.log and norm.__class__.__name__ == "LogNorm"

    # Each slider gets its own track.  vmin starts near the right of its
    # (~90%), because the starting floor is already a sensible one and what
    # you want from there is to pull it *down* and bring faint structure up
    # out of the background.  vmax starts in the middle of its own, where
    # both directions are equally reachable: raising it to stop a bright
    # bulge saturating, lowering it to bring the fainter half of the map up.
    # Both tracks are the same length, so a step is worth the same on each.
    if log:
        init_min, init_max = np.log10(norm.vmin), np.log10(norm.vmax)
        drop, head = args.decades + 3.0, 1.0        # decades of travel / of slack
        fmt = lambda v: f"{10.0 ** v:.3g}"
    else:
        span = float(norm.vmax) - float(norm.vmin)
        init_min, init_max = float(norm.vmin), float(norm.vmax)
        drop, head = span, 0.1 * span
        fmt = lambda v: f"{v:.3g}"

    fig = plt.figure(figsize=(FIG_SIZE[0], FIG_SIZE[1] + 1.4), dpi=FIG_DPI)
    ax = fig.add_axes([0.02, 0.30, 0.96, 0.64], projection="mollview")
    _, img = make_figure(values, model, unit, title, norm, args.cmap, ax=ax)

    ax_min = fig.add_axes([0.15, 0.14, 0.70, 0.04])
    ax_max = fig.add_axes([0.15, 0.08, 0.70, 0.04])
    ax_save = fig.add_axes([0.15, 0.015, 0.16, 0.05])
    ax_reset = fig.add_axes([0.34, 0.015, 0.16, 0.05])

    label = "log10 vmin" if log else "vmin"
    slider_min = Slider(ax_min, label, init_min - drop, init_min + head,
                        valinit=init_min)
    half = (drop + head) / 2.0
    slider_max = Slider(ax_max, label.replace("min", "max"),
                        init_max - half, init_max + half, valinit=init_max)
    slider_min.valtext.set_text(fmt(init_min))
    slider_max.valtext.set_text(fmt(init_max))

    def update(_):
        """Push the slider positions onto the image's norm.

        The floor is held strictly below the ceiling rather than letting the
        sliders cross: LogNorm raises on vmin >= vmax, which in a callback
        means an exception per mouse-motion event and a wedged window.
        """

        v_min, v_max = slider_min.val, slider_max.val
        if v_min >= v_max:
            v_min = v_max - (0.05 if log else max(abs(v_max), 1.0) * 1e-3)

        img.norm.vmin = 10.0 ** v_min if log else v_min
        img.norm.vmax = 10.0 ** v_max if log else v_max

        slider_min.valtext.set_text(fmt(v_min))
        slider_max.valtext.set_text(fmt(v_max))

        # Older matplotlib does not repaint the colour bar's ticks by itself
        # when the norm changes underneath it.
        if getattr(img, "colorbar", None) is not None:
            img.colorbar.update_normal(img)
        fig.canvas.draw_idle()

    slider_min.on_changed(update)
    slider_max.on_changed(update)

    def save(_):
        """Write the map at the limits currently on the sliders.

        Drawn fresh rather than cropped out of this window: the controls are
        part of this figure, and a bare Figure with its own Agg canvas is
        also what keeps pyplot from opening a second window on top of this
        one.  The result is byte-for-byte the file that
        ``plot --vmin ... --vmax ...`` would have written.
        """

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        vmin, vmax = float(img.norm.vmin), float(img.norm.vmax)

        saved = Figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        FigureCanvasAgg(saved)
        saved_ax = saved.add_subplot(111, projection="mollview")
        make_figure(values, model, unit, title,
                    sky_norm(values, vmin, vmax, args.log, args.decades),
                    args.cmap, ax=saved_ax)

        out.parent.mkdir(parents=True, exist_ok=True)
        saved.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        print(f"Wrote {out} at --vmin {vmin:.4g} --vmax {vmax:.4g}")

    def reset(_):
        slider_min.reset()
        slider_max.reset()

    button_save = Button(ax_save, "Save")
    button_reset = Button(ax_reset, "Reset")
    button_save.on_clicked(save)
    button_reset.on_clicked(reset)

    plt.show()          # blocks until the window is closed

    return float(img.norm.vmin), float(img.norm.vmax)


# ===========================================================================
# Subcommand: list
# ===========================================================================

def cmd_list(args):
    models = find_models(args.results_dir)
    print(f"{args.results_dir}: {len(models)} iterations "
          f"({models[0][0]}-{models[-1][0]})\n")

    header = f"{'iter':>5}  {'min':>12}  {'max':>12}  {'total':>12}"
    print(header)
    print("-" * len(header))

    lo, hi = np.inf, -np.inf
    unit = None
    for iteration, path in models:
        model = load_model(path)
        values = model_values(model, args.energy_index)
        unit = str(model.unit)
        lo, hi = min(lo, values.min()), max(hi, values.max())
        print(f"{iteration:5d}  {values.min():12.4g}  {values.max():12.4g}  "
              f"{values.sum():12.4g}")

    print(f"\nunit: {unit}")
    print(f"across all iterations: min {lo:.4g}, max {hi:.4g}")
    print(f"e.g.  --vmin {max(lo, hi / 1e4):.3g} --vmax {hi:.3g}")


# ===========================================================================
# Subcommand: plot
# ===========================================================================

def cmd_plot(args):
    models = dict(find_models(args.results_dir))
    available = sorted(models)
    iteration = resolve_iteration(args.iteration, available)

    use_agg(args.window)
    import matplotlib.pyplot as plt

    model = load_model(models[iteration])
    values = model_values(model, args.energy_index)
    unit = str(model.unit)

    norm = sky_norm(values, args.vmin, args.vmax, args.log, args.decades)

    fig, img = make_figure(values, model, unit,
                           frame_title(iteration, model, args),
                           norm, args.cmap)

    print(f"iteration {iteration}: data {values.min():.4g} to "
          f"{values.max():.4g} {unit}; colour scale "
          f"{norm.vmin:.4g} to {norm.vmax:.4g} "
          f"({'log' if args.log else 'linear'})")

    # Written before anything is displayed, so the file exists whether or
    # not a viewer or a GUI backend does.
    out = Path(args.out) if args.out else \
        Path(args.results_dir) / f"iteration{iteration:03d}_replot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"Wrote {out}")

    if args.window:
        plt.close(fig)          # the interactive one is built with controls
        chosen_min, chosen_max = interactive_window(
            values, model, unit, frame_title(iteration, model, args),
            norm, args, out)
        print(f"Limits on exit:  --vmin {chosen_min:.4g} "
              f"--vmax {chosen_max:.4g}")
        return

    if args.show:
        open_in_viewer(out)

    plt.close(fig)


# ===========================================================================
# Subcommand: gif
# ===========================================================================

def cmd_gif(args):
    from PIL import Image

    models = dict(find_models(args.results_dir))
    available = sorted(models)
    iterations = parse_iterations(args.iterations, available)

    if not iterations:
        raise SystemExit("No iterations selected")

    use_agg(False)
    import matplotlib.pyplot as plt

    print(f"Loading {len(iterations)} model(s)...")
    loaded = [(i, load_model(models[i])) for i in iterations]
    maps = [model_values(model, args.energy_index) for _, model in loaded]
    unit = str(loaded[-1][1].unit)

    # Every frame is read before any is drawn: a shared colour scale needs
    # the range over the whole animation, which is not known until then.
    all_values = np.concatenate(maps)

    if args.save_frames:
        frame_dir = Path(args.save_frames)
        frame_dir.mkdir(parents=True, exist_ok=True)
    else:
        frame_dir = None

    # A shared scale is one norm object for every frame -- that is what
    # makes the frames and the single colour bar mean the same thing.
    shared_norm = norm_for(args, all_values, None) if args.scale == "shared" else None

    frames, size = [], None
    for (iteration, model), values in zip(loaded, maps):
        norm = shared_norm if shared_norm is not None else \
            norm_for(args, all_values, values)

        # With differing scales the peak is the only way to tell frames
        # apart, so it goes in the title; with a shared one it is noise.
        note = "" if args.scale == "shared" else f"(peak {values.max():.3g})"
        fig, _ = make_figure(values, model, unit,
                             frame_title(iteration, model, args, note),
                             norm, args.cmap)

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=FIG_DPI)
        if frame_dir is not None:
            fig.savefig(frame_dir / f"iteration{iteration:03d}.png", dpi=FIG_DPI)
        plt.close(fig)

        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

        # The layout is fixed, so the frames should already agree; resize
        # anything that does not rather than writing a corrupt GIF.
        if size is None:
            size = image.size
        elif image.size != size:
            image = image.resize(size)

        frames.append(image)

    duration = args.duration if args.duration is not None else 1000.0 / args.fps

    # Hold the last iteration a little longer -- otherwise the animation
    # loops straight off the converged image, which is the one worth seeing.
    durations = [duration] * len(frames)
    durations[-1] = duration * max(1.0, args.hold)

    out = Path(args.out) if args.out else Path(args.results_dir) / "iterations.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=[int(d) for d in durations],
                   loop=0 if args.loop else 1, optimize=True)

    scale_note = (f"shared scale {shared_norm.vmin:.4g} to "
                  f"{shared_norm.vmax:.4g} {unit}"
                  if shared_norm is not None else "autoscaled per frame")
    print(f"Wrote {out}: {len(frames)} frames, {duration:.0f} ms each, "
          f"{scale_note}, {'log' if args.log else 'linear'}")
    if frame_dir is not None:
        print(f"Frames also written to {frame_dir}")


# ===========================================================================
# Command line
# ===========================================================================

def add_common(parser):
    parser.add_argument("results_dir", nargs="?", default=None,
                        help="results directory, e.g. "
                             "results/ns32_e505-517_n313368_bkg75000s12345 "
                             "(default: the most recent run under results/)")
    parser.add_argument("--energy-index", type=int, default=0,
                        help="which energy bin of the model to plot "
                             "(default: 0)")


def add_scale(parser):
    parser.add_argument("--vmin", type=float, default=None,
                        help="bottom of the colour scale (default: from data)")
    parser.add_argument("--vmax", type=float, default=None,
                        help="top of the colour scale (default: from data)")
    parser.add_argument("--linear", dest="log", action="store_false",
                        help="linear colour scale (default: logarithmic)")
    parser.add_argument("--log", dest="log", action="store_true",
                        help="logarithmic colour scale (the default)")
    parser.set_defaults(log=True)
    parser.add_argument("--decades", type=float, default=6.0,
                        help="how many decades below the peak a log scale "
                             "reaches when --vmin is not given, and how far "
                             "the --window sliders travel (default: 6).  "
                             "e.g. --decades 3 on a map peaking at 1e-2 puts "
                             "the floor at 1e-5, so only the brightest three "
                             "decades get colour and the faint background "
                             "goes flat; --decades 9 pulls that floor to "
                             "1e-11 and shows the halo instead")
    parser.add_argument("--cmap", default="viridis",
                        help="matplotlib colormap (default: viridis)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run a subcommand with --help for its options.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_list = subparsers.add_parser(
        "list", help="show the iterations and their value ranges")
    add_common(p_list)
    p_list.set_defaults(func=cmd_list)

    p_plot = subparsers.add_parser(
        "plot", help="plot one iteration as a mollview")
    add_common(p_plot)
    add_scale(p_plot)
    p_plot.add_argument("-i", "--iteration", type=int, default=None,
                        help="iteration to plot; negative counts back from "
                             "the last (default: the last one)")
    p_plot.add_argument("-o", "--out", default=None,
                        help="output image (default: "
                             "<results_dir>/iterationNNN_replot.png)")
    p_plot.add_argument("--no-show", dest="show", action="store_false",
                        help="just write the file, do not open it")
    p_plot.add_argument("--window", action="store_true",
                        help="show it in an interactive window with vmin/vmax "
                             "sliders instead of an image viewer; blocks "
                             "until you close it, then prints the limits you "
                             "landed on")
    p_plot.set_defaults(func=cmd_plot, show=True, window=False)

    p_gif = subparsers.add_parser(
        "gif", help="stitch the iterations into an animated GIF")
    add_common(p_gif)
    add_scale(p_gif)
    p_gif.add_argument("--iterations", default=None,
                       help="which iterations, e.g. 1-30, 1-50:5, or 1,5,9 "
                            "(default: all of them)")
    p_gif.add_argument("--scale", choices=("shared", "per-frame"),
                       default="shared",
                       help="one colour scale for every frame, or autoscale "
                            "each (default: shared).  --vmin/--vmax override "
                            "either way")
    p_gif.add_argument("--fps", type=float, default=4.0,
                       help="frames per second (default: 4)")
    p_gif.add_argument("--duration", type=float, default=None,
                       help="milliseconds per frame; overrides --fps")
    p_gif.add_argument("--hold", type=float, default=4.0,
                       help="hold the last frame this many times longer "
                            "(default: 4)")
    p_gif.add_argument("--no-loop", dest="loop", action="store_false",
                       help="play once instead of looping forever")
    p_gif.add_argument("-o", "--out", default=None,
                       help="output GIF (default: "
                            "<results_dir>/iterations.gif)")
    p_gif.add_argument("--save-frames", default=None,
                       help="also write the individual frames to this "
                            "directory")
    p_gif.set_defaults(func=cmd_gif, loop=True)

    args = parser.parse_args(argv)

    if args.results_dir is None:
        args.results_dir = default_results_dir()
        print(f"Using {args.results_dir}")

    args.func(args)


if __name__ == "__main__":
    main()
