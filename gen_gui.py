#!/usr/bin/env python3
"""Viewer for generated MNIST samples, built on the helpers in cnn_gui.

Models are plugged in through plain callables

    Sampler = Callable[[], array]
        () -> (N, 28, 28) or (N, 1, 28, 28) images in [0, 1], fresh noise each call
    BatchPredictor = Callable[[np.ndarray], np.ndarray]
        (N, 1, 28, 28) images -> (N, 10) class probabilities

`eqx_sampler` / `eqx_batch_predictor` build them from equinox models.

Unlike `cnn_gui.show_images`, pixels are shown on a fixed [0, 1] scale rather
than rescaled per image, so faint or washed-out samples look the way they are.
"""

from collections import Counter
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from cnn_gui import MNIST_SIZE, TEXT_COLOR, WRONG_COLOR, to_numpy_images

Sampler = Callable[[], np.ndarray]
BatchPredictor = Callable[[np.ndarray], np.ndarray]

LOW_CONFIDENCE = 0.5


# Model adapters


def eqx_sampler(generator, z_dim: int, n: int = 64) -> Sampler:
    """Wrap an equinox generator mapping (z_dim,) noise -> (1, 28, 28) image."""
    import equinox as eqx
    import jax

    import jax_utils

    @eqx.filter_jit
    def generate(generator, z):
        return jax.vmap(generator)(z)

    def sample() -> np.ndarray:
        z = jax.random.normal(jax_utils.get_key(), (n, z_dim))
        return np.asarray(generate(generator, z))

    return sample


def eqx_batch_predictor(model) -> BatchPredictor:
    """Wrap an equinox classifier mapping (1, 28, 28) -> 10 logits or log-probs."""
    import equinox as eqx
    import jax

    @eqx.filter_jit
    def probs(model, x):
        return jax.nn.softmax(jax.vmap(model)(x))

    def predict(images: np.ndarray) -> np.ndarray:
        x = np.asarray(images, np.float32).reshape(-1, 1, MNIST_SIZE, MNIST_SIZE)
        return np.asarray(probs(model, x))

    return predict


# Viewer


def launch_sample_viewer(
    sample: Sampler,
    predict_batch: Optional[BatchPredictor] = None,
    ncols: int = 8,
    block: bool = True,
):
    """Open a window with a grid of samples and a Resample button (or press r).

    With `predict_batch`, each sample is titled with its predicted class and
    confidence (red below LOW_CONFIDENCE), and the window title counts the
    predicted classes, which makes mode collapse easy to spot.

    Returns the figure; keep a reference to it, or the button stops working.
    """
    images = to_numpy_images(sample())
    n = len(images)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)

    fig = plt.figure(figsize=(1.2 * ncols, 1.35 * nrows + 1))
    grid = fig.add_gridspec(nrows + 1, ncols, height_ratios=[10] * nrows + [4])
    axes = []
    for i in range(n):
        ax = fig.add_subplot(grid[i // ncols, i % ncols])
        ax.set_axis_off()
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes.append(ax)

    def show(images: np.ndarray) -> None:
        for ax, image in zip(axes, images):
            ax.images[0].set_data(image)
        if predict_batch is None:
            fig.suptitle("generated samples", color=TEXT_COLOR)
        else:
            probs = predict_batch(images)
            preds, confs = probs.argmax(-1), probs.max(-1)
            for ax, pred, conf in zip(axes, preds, confs):
                color = WRONG_COLOR if conf < LOW_CONFIDENCE else TEXT_COLOR
                ax.set_title(f"{pred} ({conf:.2f})", fontsize=9, color=color)
            counts = Counter(preds.tolist())
            hist = "  ".join(f"{d}:{counts.get(d, 0)}" for d in range(10))
            fig.suptitle(
                f"predicted classes  {hist}\nmean confidence {confs.mean():.2f}",
                color=TEXT_COLOR,
                fontsize=10,
            )
        fig.canvas.draw_idle()

    def resample(_=None) -> None:
        show(to_numpy_images(sample()))

    button = Button(fig.add_subplot(grid[nrows, : min(2, ncols)]), "Resample")
    button.on_clicked(resample)
    fig.resample_button = button  # widgets die if nothing references them
    fig.canvas.mpl_connect(
        "key_press_event", lambda event: event.key == "r" and resample()
    )

    show(images)
    fig.tight_layout()
    if block:
        plt.show()
    return fig
