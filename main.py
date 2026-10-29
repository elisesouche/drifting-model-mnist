#!/usr/bin/env python3
from dataclasses import dataclass
import typer
import cnn

app = typer.Typer()


class CliArgs:
    name: str


def main(cli: CliArgs):
    print(f"hello, {cli.name}")


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


DRIFT_PATH = "./drift.eqx"


def init_generator():
    import drift
    import jax_utils

    return drift.ConvGenerator(jax_utils.get_key())


@app.command()
def train_drift(path: str = DRIFT_PATH, steps: int = 5000):
    """Train the generator with the drifting loss."""
    import equinox as eqx
    import drift

    generator = drift.train(init_generator(), steps, cnn.deser_model(CNN_PATH))
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
