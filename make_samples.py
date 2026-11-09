#!/usr/bin/env python3
"""Train one generator in pixel space and one in CNN feature space, and save a
grid of samples for each. Both runs share generator, seed and temperature."""

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch

import cnn
import drift
import jax_utils

STEPS = 15000
NROWS, NCOLS = 4, 8
OUT_DIR = "../extended_abstract/figures"


class PixelEncoder(eqx.Module):
    """Single view of the raw pixels, so that `drift.feature_loss` normalizes
    pixel distances exactly as it does feature distances."""

    def features(self, x):
        return [x]


def reset_seed():
    jax_utils.__key = None
    torch.manual_seed(jax_utils.SEED)


def save_grid(samples, path):
    rows = [np.hstack(samples[i * NCOLS : (i + 1) * NCOLS]) for i in range(NROWS)]
    plt.imsave(path, np.vstack(rows), cmap="gray", vmin=0, vmax=1)


def run(name, encoder):
    reset_seed()
    generator = drift.UpsampleGenerator(jax_utils.get_key())
    generator = drift.train(generator, STEPS, encoder)
    eqx.tree_serialise_leaves(f"./samples_{name}.eqx", generator)

    z = jax.random.normal(jax.random.key(0), (NROWS * NCOLS, drift.Z_DIM))
    samples = np.asarray(jax.vmap(generator)(z)).squeeze(1)
    save_grid(samples, f"{OUT_DIR}/samples_{name}.png")


if __name__ == "__main__":
    run("pixel", PixelEncoder())
    run("feature", cnn.deser_model("./cnn.eqx"))
