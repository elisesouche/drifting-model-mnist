#!/usr/bin/env python3
"""Measure the defect rates of params.md on real MNIST and on saved generators.

pieces: share of images whose ink (> 0.5) forms more than one component.
specks: share of images with a component of fewer than 6 pixels.
"""

import equinox as eqx
import jax
import numpy as np
from scipy import ndimage
import torch

import drift
import jax_utils
import mnist

N = 1024
CHECKPOINTS = ["samples_pixel.eqx", "samples_feature.eqx", "drift.eqx"]


def defects(images):
    pieces, specks = [], []
    for ink in images > 0.5:
        labels, count = ndimage.label(ink)
        pieces.append(count > 1)
        specks.append(np.any(np.bincount(labels.ravel())[1:] < 6))
    return np.mean(pieces), np.mean(specks)


def report(name, images):
    pieces, specks = defects(images)
    print(f"{name}: pieces = {pieces:.1%}, specks = {specks:.1%}")


if __name__ == "__main__":
    torch.manual_seed(jax_utils.SEED)
    real = next(iter(mnist.get_loader(True, N, augment=False)))[0].numpy()
    report("real", real[:, 0])

    z = jax.random.normal(jax.random.key(1), (N, drift.Z_DIM))
    for path in CHECKPOINTS:
        template = drift.UpsampleGenerator(jax.random.key(0))
        generator = eqx.tree_deserialise_leaves(path, template)
        report(path, np.asarray(jax.vmap(generator)(z))[:, 0])
