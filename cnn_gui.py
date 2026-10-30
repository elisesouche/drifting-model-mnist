#!/usr/bin/env python3
"""Rendering and interactive drawing utilities for MNIST experiments.

Everything here is framework agnostic: images can be numpy arrays, jax arrays
or torch tensors, and models are plugged in through a plain callable

    Predictor = Callable[[np.ndarray], np.ndarray]
        (28, 28) float image in [0, 1], ink = 1  ->  (10,) class probabilities

`eqx_predictor` builds one from an equinox model such as the CNN in scratch.py.

Pieces:
    show_images / show_loader_batch   grid of digits with labels/predictions
    plot_probs                        bar chart of class probabilities
    DrawingPad                        mouse-drawable 28x28 canvas, emits images
    ProbabilityView                   subscribes to a pad, shows a predictor's output
    InputView                         subscribes to a pad, shows what the model sees
    centre_like_mnist                 crop / rescale / centre a drawing like MNIST
    launch_drawing_app                pad + views wired together in one window

Run `python mnistutils.py` to train scratch.py's CNN and draw digits for it.
"""

import time
from typing import Callable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from PIL import Image

Predictor = Callable[[np.ndarray], np.ndarray]
ImageCallback = Callable[[np.ndarray], None]

MNIST_SIZE = 28
INK_COLOR = "#2a6fdb"
MUTED_COLOR = "#b8bcc4"
TEXT_COLOR = "#3b3f46"
WRONG_COLOR = "#c8323c"


# Conversion


def to_numpy_images(images) -> np.ndarray:
    """Convert any (28,28), (N,28,28), (1,28,28) or (N,1,28,28) array-like to (N,28,28)."""
    if hasattr(images, "detach"):  # torch tensor
        images = images.detach().cpu().numpy()
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 2:
        images = images[None]
    elif images.ndim == 4:
        images = images[:, 0]
    if images.ndim != 3:
        raise ValueError(f"Cannot interpret shape {images.shape} as MNIST images")
    return images


def to_numpy_labels(labels) -> Optional[np.ndarray]:
    if labels is None:
        return None
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


# Static rendering


def show_images(
    images,
    labels=None,
    preds=None,
    ncols: int = 8,
    title: Optional[str] = None,
    cmap: str = "gray",
    fig: Optional[Figure] = None,
) -> Figure:
    """Show a grid of digits, each titled with its label and/or prediction.

    Images are rescaled per image for display, so both [0, 1] and normalised
    [-1, 1] inputs render correctly. Wrong predictions are titled in red.
    """
    images = to_numpy_images(images)
    labels = to_numpy_labels(labels)
    preds = to_numpy_labels(preds)
    n = len(images)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)

    if fig is None:
        fig = plt.figure(figsize=(1.2 * ncols, 1.35 * nrows + (0.4 if title else 0)))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i >= n:
            continue
        ax.imshow(images[i], cmap=cmap, interpolation="nearest")
        parts, color = [], TEXT_COLOR
        if labels is not None:
            parts.append(f"y={labels[i]}")
        if preds is not None:
            parts.append(f"ŷ={preds[i]}")
            if labels is not None and preds[i] != labels[i]:
                parts[-1] += " ✗"
                color = WRONG_COLOR
        if parts:
            ax.set_title(" ".join(parts), fontsize=9, color=color)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_loader_batch(
    loader,
    n: int = 32,
    predict_batch: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    **kwargs,
) -> Figure:
    """Show the first `n` images of the next batch of a (torch) DataLoader.

    `predict_batch` maps the batch exactly as the loader yields it, as numpy
    (N, 1, 28, 28), to (N,) predicted labels, e.g.
    `lambda x: jax.vmap(model)(x).argmax(-1)`.
    """
    x, y = next(iter(loader))
    x = x[:n].numpy() if hasattr(x, "numpy") else np.asarray(x[:n])
    preds = None if predict_batch is None else np.asarray(predict_batch(x))
    return show_images(x, labels=y[:n], preds=preds, **kwargs)


def plot_probs(ax: Axes, probs, true_label: Optional[int] = None):
    """Bar chart of class probabilities on `ax`; returns the bar container.

    Use `update_probs` to refresh it in place (cheap enough for live updates).
    """
    ax.clear()
    bars = ax.bar(range(10), np.zeros(10), width=0.6, color=MUTED_COLOR)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1.1)  # headroom for the value label
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(colors=TEXT_COLOR, length=0)
    ax.grid(axis="y", color="#e6e8eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED_COLOR)
    ax.set_xlabel("digit", color=TEXT_COLOR)
    ax.set_ylabel("probability", color=TEXT_COLOR)
    bars.value_text = ax.text(0, 0, "", ha="center", va="bottom", color=TEXT_COLOR)
    bars.true_label = true_label
    update_probs(bars, probs)
    return bars


def update_probs(bars, probs) -> None:
    probs = np.asarray(probs, dtype=np.float32).reshape(10)
    best = int(probs.argmax())
    for i, bar in enumerate(bars):
        bar.set_height(probs[i])
        bar.set_color(INK_COLOR if i == best else MUTED_COLOR)
        if getattr(bars, "true_label", None) == i:
            bar.set_edgecolor(TEXT_COLOR)
            bar.set_linewidth(1.5)
    bars.value_text.set_position((best, probs[best] + 0.02))
    bars.value_text.set_text(f"{probs[best]:.2f}")


# Preprocessing


def centre_like_mnist(image: np.ndarray, box: int = 20) -> np.ndarray:
    """Normalise a free-hand drawing the way MNIST digits were prepared:
    crop to the ink, rescale so the longest side is `box` px (aspect kept),
    then place it in a 28x28 frame with its centre of mass in the middle.
    """
    image = np.asarray(image, dtype=np.float32)
    rows = np.flatnonzero(image.max(axis=1) > 0.05)
    cols = np.flatnonzero(image.max(axis=0) > 0.05)
    if len(rows) == 0:
        return np.zeros((MNIST_SIZE, MNIST_SIZE), np.float32)
    crop = image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    h, w = crop.shape
    scale = box / max(h, w)
    new_h, new_w = max(1, round(h * scale)), max(1, round(w * scale))
    resized = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit = np.clip(np.asarray(resized, dtype=np.float32), 0, 1)

    ys, xs = np.indices(digit.shape)
    mass = digit.sum()
    cy, cx = (ys * digit).sum() / mass, (xs * digit).sum() / mass
    top = int(round(MNIST_SIZE / 2 - cy))
    left = int(round(MNIST_SIZE / 2 - cx))

    # paste into an oversized canvas so shifts can never index out of bounds
    pad = MNIST_SIZE
    canvas = np.zeros((3 * MNIST_SIZE, 3 * MNIST_SIZE), np.float32)
    canvas[pad + top : pad + top + new_h, pad + left : pad + left + new_w] = digit
    return canvas[pad : pad + MNIST_SIZE, pad : pad + MNIST_SIZE]


# Model adapters


def eqx_predictor(model, mean: float = 0, std: float = 1) -> Predictor:
    """Wrap an equinox model mapping (1,28,28) -> 10 logits or log-probs.

    `mean`/`std` must match the Normalize transform used during training.
    """
    import equinox as eqx
    import jax

    @eqx.filter_jit
    def probs(model, x):
        return jax.nn.softmax(model(x))  # also correct for log_softmax outputs

    def predict(image: np.ndarray) -> np.ndarray:
        x = (np.asarray(image, np.float32)[None] - mean) / std
        return np.asarray(probs(model, x))

    return predict


def eqx_reconstructor(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an equinox autoencoder mapping (1,28,28) -> (1,28,28) logits."""
    import equinox as eqx
    import jax

    @eqx.filter_jit
    def reconstruct(model, x):
        return jax.nn.sigmoid(model(x))

    def run(image: np.ndarray) -> np.ndarray:
        x = np.asarray(image, np.float32)[None]
        return np.asarray(reconstruct(model, x))[0]

    return run


# Interactive drawing


class BlitManager:
    """Fast partial redraws for one figure.

    A full redraw of a figure with a few axes takes tens of milliseconds, far
    too slow to follow the mouse. Registered artists are instead drawn on top
    of a cached background (axes, ticks, labels), which is only re-rendered
    when the figure itself is redrawn (e.g. on resize).
    """

    def __init__(self, canvas):
        self.canvas = canvas
        self._artists = []
        self._background = None
        canvas.mpl_connect("draw_event", self._on_draw)

    def add(self, *artists) -> None:
        for artist in artists:
            artist.set_animated(True)  # excluded from normal redraws
            self._artists.append(artist)

    def _on_draw(self, event) -> None:
        self._background = self.canvas.copy_from_bbox(self.canvas.figure.bbox)
        self._draw_artists()

    def _draw_artists(self) -> None:
        for artist in self._artists:
            self.canvas.figure.draw_artist(artist)

    def update(self) -> None:
        if self._background is None:  # first draw has not happened yet
            self.canvas.draw_idle()
            return
        self.canvas.restore_region(self._background)
        self._draw_artists()
        # push only the changed axes to the screen, not the whole window
        for ax in {artist.axes for artist in self._artists}:
            self.canvas.blit(ax.bbox)
        # no flush_events() here: it would run queued mouse events re-entrantly
        # in the middle of the current one


class DrawingPad:
    """A mouse-drawable digit canvas living on a matplotlib Axes.

    Left button draws, right button erases. Strokes are rasterised on an
    `upscale`x finer grid and area-averaged down to 28x28, which gives the soft
    anti-aliased edges of real MNIST digits. Every change is pushed to the
    subscribed callbacks as a (28, 28) float array in [0, 1].

    Mouse events only paint; callbacks and repaints are rate limited to
    `max_fps`, with a trailing refresh so the final state is always shown.
    With a `blitter`, only artists registered on it are repainted (fast);
    views on the same figure should register theirs too. Without one, the
    whole figure is redrawn.
    """

    def __init__(
        self,
        ax: Axes,
        brush_radius: float = 1.3,
        upscale: int = 8,
        callbacks: Sequence[ImageCallback] = (),
        blitter: Optional[BlitManager] = None,
        max_fps: float = 60,
    ):
        self.ax = ax
        self.blitter = blitter
        self.min_interval = 1 / max_fps
        self._last_refresh = 0.0
        self._refresh_timer = ax.figure.canvas.new_timer()
        self._refresh_timer.single_shot = True
        self._refresh_timer.add_callback(self._refresh)
        self._refresh_pending = False
        self.upscale = upscale
        self.brush_radius = brush_radius * upscale
        self._hires = np.zeros((MNIST_SIZE * upscale,) * 2, np.float32)
        self._callbacks: list[ImageCallback] = list(callbacks)
        self._last: Optional[tuple[float, float]] = None
        self._erasing = False

        self._artist = ax.imshow(
            self.image, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("draw here  (right click erases)", fontsize=10, color=TEXT_COLOR)
        self.artists = [self._artist]
        if blitter is not None:
            blitter.add(*self.artists)

        canvas = ax.figure.canvas
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    @property
    def image(self) -> np.ndarray:
        u = self.upscale
        return self._hires.reshape(MNIST_SIZE, u, MNIST_SIZE, u).mean(axis=(1, 3))

    def subscribe(self, callback: ImageCallback) -> Callable[[], None]:
        """Register `callback(image)`; returns a function that unsubscribes it."""
        self._callbacks.append(callback)
        callback(self.image)
        return lambda: self._callbacks.remove(callback)

    def clear(self) -> None:
        self._hires[:] = 0
        self._refresh()

    def set_image(self, image) -> None:
        """Load an existing 28x28 image (e.g. a dataset sample) to edit it."""
        image = to_numpy_images(image)[0]
        lo, hi = image.min(), image.max()
        image = (image - lo) / (hi - lo) if hi > lo else np.zeros_like(image)
        self._hires[:] = np.kron(image, np.ones((self.upscale, self.upscale)))
        self._refresh()

    def flush(self) -> None:
        """Run a pending rate-limited refresh now (e.g. before savefig)."""
        if self._refresh_pending:
            self._refresh()

    def disconnect(self) -> None:
        for cid in self._cids:
            self.ax.figure.canvas.mpl_disconnect(cid)

    def _to_hires(self, event) -> tuple[float, float]:
        # imshow pixel centres sit on integer data coordinates
        u = self.upscale
        return (event.xdata + 0.5) * u, (event.ydata + 0.5) * u

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax or event.button not in (1, 3):
            return
        self._erasing = event.button == 3
        self._last = self._to_hires(event)
        self._stroke(self._last, self._last)

    def _on_motion(self, event) -> None:
        if self._last is None or event.inaxes is not self.ax:
            return
        start, self._last = self._last, self._to_hires(event)
        self._stroke(start, self._last)

    def _on_release(self, event) -> None:
        self._last = None
        self.flush()

    def _stroke(self, start, end) -> None:
        """Paint a capsule (thick line segment with round caps) from start to end."""
        r = self.brush_radius * (2 if self._erasing else 1)
        (x0, y0), (x1, y1) = start, end
        n = self._hires.shape[0]
        lo_x, hi_x = int(max(0, min(x0, x1) - r)), int(min(n, max(x0, x1) + r + 1))
        lo_y, hi_y = int(max(0, min(y0, y1) - r)), int(min(n, max(y0, y1) + r + 1))
        if lo_x >= hi_x or lo_y >= hi_y:
            return
        ys, xs = np.mgrid[lo_y:hi_y, lo_x:hi_x] + 0.5
        dx, dy = x1 - x0, y1 - y0
        length2 = dx * dx + dy * dy
        t = (
            0.0
            if length2 == 0
            else np.clip(((xs - x0) * dx + (ys - y0) * dy) / length2, 0, 1)
        )
        dist2 = (xs - (x0 + t * dx)) ** 2 + (ys - (y0 + t * dy)) ** 2
        mask = dist2 <= r * r
        self._hires[lo_y:hi_y, lo_x:hi_x][mask] = 0.0 if self._erasing else 1.0
        self._changed()

    def _changed(self) -> None:
        wait = self.min_interval - (time.perf_counter() - self._last_refresh)
        if wait <= 0:
            self._refresh()
        elif not self._refresh_pending:
            self._refresh_pending = True
            self._refresh_timer.interval = int(wait * 1000) + 1
            self._refresh_timer.start()

    def _refresh(self) -> None:
        self._refresh_timer.stop()
        self._refresh_pending = False
        image = self.image
        self._artist.set_data(image)
        for callback in self._callbacks:
            callback(image)
        if self.blitter is not None:
            self.blitter.update()
        else:
            self.ax.figure.canvas.draw_idle()
        # measured from the end, so queued mouse events get handled in between
        # refreshes even when a refresh takes longer than min_interval
        self._last_refresh = time.perf_counter()


class InputView:
    """Shows an image after `preprocess`, i.e. exactly what the model receives."""

    def __init__(
        self,
        ax: Axes,
        preprocess: Callable[[np.ndarray], np.ndarray] = centre_like_mnist,
        title: str = "model input",
    ):
        self.preprocess = preprocess
        self._artist = ax.imshow(
            np.zeros((MNIST_SIZE, MNIST_SIZE)),
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, color=TEXT_COLOR)
        self.artists = [self._artist]

    def __call__(self, image: np.ndarray) -> None:
        self._artist.set_data(self.preprocess(image))


class ProbabilityView:
    """Shows `predict(preprocess(image))` as a live bar chart."""

    def __init__(
        self,
        ax: Axes,
        predict: Predictor,
        preprocess: Callable[[np.ndarray], np.ndarray] = centre_like_mnist,
    ):
        self.ax = ax
        self.predict = predict
        self.preprocess = preprocess
        self._bars = plot_probs(ax, np.zeros(10))
        ax.set_title("prediction", fontsize=10, color=TEXT_COLOR)
        self.artists = [*self._bars, self._bars.value_text]

    def __call__(self, image: np.ndarray) -> None:
        if image.max() < 0.05:  # blank canvas: nothing to classify
            update_probs(self._bars, np.zeros(10))
            self._bars.value_text.set_text("")
            return
        update_probs(self._bars, self.predict(self.preprocess(image)))


def launch_drawing_app(
    predict: Optional[Predictor] = None,
    preprocess: Callable[[np.ndarray], np.ndarray] = centre_like_mnist,
    initial_image=None,
    block: bool = True,
    reconstruct: Optional[Callable[[np.ndarray], np.ndarray]] = None,
):
    """Open a window with a drawing pad, the preprocessed model input and,
    if given, the live class probabilities from `predict` and the image
    rebuilt by `reconstruct` (e.g. an autoencoder, see `eqx_reconstructor`).

    Returns (fig, pad); keep a reference to `pad`, or its callbacks get
    garbage collected.
    """
    widths = [3, 1.6]
    if reconstruct is not None:
        widths.append(1.6)
    if predict is not None:
        widths.append(3.4)
    fig = plt.figure(figsize=(sum(widths) + 2, 4))
    grid = fig.add_gridspec(2, len(widths), width_ratios=widths, height_ratios=[10, 1])
    blitter = BlitManager(fig.canvas)
    pad = DrawingPad(fig.add_subplot(grid[0, 0]), blitter=blitter)
    views = [InputView(fig.add_subplot(grid[0, 1]), preprocess)]
    if reconstruct is not None:
        views.append(
            InputView(
                fig.add_subplot(grid[0, len(views) + 1]),
                lambda image: reconstruct(preprocess(image)),
                title="reconstruction",
            )
        )
    if predict is not None:
        views.append(
            ProbabilityView(fig.add_subplot(grid[0, len(views) + 1]), predict, preprocess)
        )
    for view in views:
        blitter.add(*view.artists)
        pad.subscribe(view)
    if initial_image is not None:
        pad.set_image(initial_image)

    button = Button(fig.add_subplot(grid[1, 0]), "Clear")
    button.on_clicked(lambda _: pad.clear())
    pad.clear_button = button  # widgets die if nothing references them

    fig.tight_layout()
    if block:
        plt.show()
    return fig, pad


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "mode",
        nargs="?",
        default="draw",
        choices=["draw", "grid", "both"],
        help="draw digits for the CNN, show a predicted batch, or both",
    )
    args = parser.parse_args()

    import jax
    import scratch  # trains the CNN on import

    if args.mode in ("grid", "both"):
        show_loader_batch(
            scratch.trainloader,
            predict_batch=lambda x: jax.vmap(scratch.model)(x).argmax(-1),
            title="training batch: label y, prediction ŷ",
        )
        if args.mode == "grid":
            plt.show()
    if args.mode in ("draw", "both"):
        launch_drawing_app(eqx_predictor(scratch.model, mean=0, std=1), lambda img: img)
