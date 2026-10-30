#!/usr/bin/env python3
from dataclasses import dataclass
import typer
import cnn

app = typer.Typer()


CNN_PATH = "./cnn.eqx"


@app.command()
def train_cnn(path: str = CNN_PATH, epochs: int = 5):
    """Train the feature encoder and classifier."""
    model = cnn.train_model(epochs)
    cnn.ser_model(model, path)


@app.command()
def cnn_gui():
    import cnn_gui

    model = cnn.deser_model(CNN_PATH)
    cnn_gui.launch_drawing_app(cnn_gui.eqx_predictor(model))


CAE_PATH = "./cae.eqx"


@app.command()
def train_cae(path: str = CAE_PATH, epochs: int = 5):
    """Train the convolutional autoencoder (an alternative feature encoder)."""
    import cae

    model = cae.train_model(epochs)
    cae.ser_model(model, path)


@app.command()
def cae_gui(path: str = CAE_PATH, n: int = 16):
    """Show test digits next to their reconstructions, and a pad to draw your own."""
    import jax
    import numpy as np
    import cae
    import cnn_gui
    import mnist

    model = cae.deser_model(path)
    x, _ = next(iter(mnist.get_loader(False, n)))
    x = x.numpy()
    recon = np.asarray(jax.nn.sigmoid(jax.vmap(model)(x)))
    cnn_gui.show_images(
        np.concatenate([x, recon]),
        ncols=n,
        title="test digits (top) and their reconstructions (bottom)",
    )
    cnn_gui.launch_drawing_app(reconstruct=cnn_gui.eqx_reconstructor(model))


DRIFT_PATH = "./drift.eqx"


def load_encoder(encoder: str):
    if encoder == "cnn":
        return cnn.deser_model(CNN_PATH)
    if encoder == "cae":
        import cae

        return cae.deser_model(CAE_PATH)
    raise ValueError(f"Unknown encoder {encoder!r}, expected 'cnn' or 'cae'")


def init_generator():
    import drift
    import jax_utils

    return drift.UpsampleGenerator(jax_utils.get_key())


@app.command()
def train_drift(path: str = DRIFT_PATH, steps: int = 5000, encoder: str = "cnn"):
    """Train the generator with the drifting loss, in the features of `encoder` (cnn or cae)."""
    import equinox as eqx
    import drift

    generator = drift.train(init_generator(), steps, load_encoder(encoder))
    eqx.tree_serialise_leaves(path, generator)


@app.command()
def drift_gui(path: str = DRIFT_PATH, n: int = 64):
    """Show generated samples, labelled by the CNN."""
    import equinox as eqx
    import drift
    import gen_gui

    generator = eqx.tree_deserialise_leaves(path, init_generator())
    gen_gui.launch_sample_viewer(
        gen_gui.eqx_sampler(generator, drift.Z_DIM, n),
        gen_gui.eqx_batch_predictor(cnn.deser_model(CNN_PATH)),
    )


if __name__ == "__main__":
    app()
