#!/usr/bin/env python3
"""Compare real MNIST and generated samples through the eyes of two classifiers:
`cnn.eqx`, which defines the training features, and an independently trained one.

Real images come from the test set, which neither classifier was trained on.
"""

import os

import equinox as eqx
import jax
import numpy as np
import torch

import cnn
import drift
import jax_utils
import mnist

N = 1024
INDEPENDENT_PATH = "./cnn_eval.eqx"
INDEPENDENT_SEED = 7211


def independent_classifier():
    if not os.path.exists(INDEPENDENT_PATH):
        jax_utils.SEED = INDEPENDENT_SEED
        torch.manual_seed(INDEPENDENT_SEED)
        cnn.ser_model(cnn.train_model(5), INDEPENDENT_PATH)
    return cnn.deser_model(INDEPENDENT_PATH)


def probabilities(classifier, images):
    return np.asarray(jax.nn.softmax(jax.vmap(classifier)(images)))


def report(name, probs):
    confidence = probs.max(axis=1)
    shares = np.bincount(probs.argmax(axis=1), minlength=10) / len(probs)
    marginal = probs.mean(axis=0)
    kl = np.sum(probs * (np.log(probs + 1e-12) - np.log(marginal + 1e-12)), axis=1)
    print(
        f"  {name}: mean confidence = {confidence.mean():.3f}, "
        f"below 0.5 = {np.mean(confidence < 0.5):.1%}, "
        f"below 0.9 = {np.mean(confidence < 0.9):.1%}, "
        f"score = {np.exp(kl.mean()):.2f}"
    )
    print("    classes: " + " ".join(f"{d}:{s:.1%}" for d, s in enumerate(shares)))


if __name__ == "__main__":
    classifiers = {
        "independent": independent_classifier(),
        "cnn.eqx (encoder)": cnn.deser_model("./cnn.eqx"),
    }

    x_test, y_test = next(iter(mnist.get_loader(False, 10000, augment=False)))
    x_test, y_test = x_test.numpy(), y_test.numpy()

    z = jax.random.normal(jax.random.key(1), (N, drift.Z_DIM))
    samples = {}
    for name in ["pixel", "feature"]:
        template = drift.UpsampleGenerator(jax.random.key(0))
        generator = eqx.tree_deserialise_leaves(f"./samples_{name}.eqx", template)
        samples[name] = np.asarray(jax.vmap(generator)(z))

    for classifier_name, classifier in classifiers.items():
        print(classifier_name)
        probs = probabilities(classifier, x_test)
        print(f"  real test accuracy = {np.mean(probs.argmax(axis=1) == y_test):.2%}")
        report("real", probs)
        for name, images in samples.items():
            report(f"upsampling + {name}", probabilities(classifier, images))
