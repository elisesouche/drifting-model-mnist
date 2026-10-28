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
def greet(name: str):
    print(f"Hey {name}!")

@app.command()
def cnn_gui():
    import cnn_gui
    model = cnn.deser_model(CNN_PATH)
    cnn_gui.launch_drawing_app(cnn_gui.eqx_predictor(model))


if __name__ == "__main__":
    app()
