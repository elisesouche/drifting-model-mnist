#!/usr/bin/env python3

import jax
import equinox as eqx
from jaxtyping import Float, Array
import optax
import jax_utils

LATENT_CHANNELS = 1
BATCH_SIZE = 64
LEARNING_RATE = 1e-3


class CAE(eqx.Module):
    encoder: list
    decoder: list

    FEATURE_LAYERS = (1, 4, 7)

    def __init__(self, rand_key):
        super().__init__()
        k1, k2, k3, k4, k5, k6 = jax.random.split(rand_key, 6)
        self.encoder = [
            eqx.nn.Conv2d(1, 64, kernel_size=3, padding=1, key=k1),  # 64×28×28
            jax.nn.silu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),  # 64×14×14
            eqx.nn.Conv2d(64, 128, kernel_size=3, padding=1, key=k2),  # 128×14×14
            jax.nn.silu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),  # 128×7×7
            eqx.nn.Conv2d(
                128, LATENT_CHANNELS, kernel_size=3, padding=1, key=k3
            ),  # 16×7×7
            jax.nn.silu,
        ]
        self.decoder = [
            eqx.nn.ConvTranspose2d(
                LATENT_CHANNELS, 128, 4, stride=2, padding=1, key=k4
            ),  # 128×14×14
            jax.nn.silu,
            eqx.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, key=k5),  # 64×28×28
            jax.nn.silu,
            eqx.nn.Conv2d(64, 1, kernel_size=3, padding=1, key=k6),  # 1×28×28, logits
        ]

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "1 28 28"]:
        """Reconstruction logits; apply jax.nn.sigmoid to get an image."""
        for layer in self.encoder + self.decoder:
            x = layer(x)
        return x

    def features(self, x: Float[Array, "1 28 28"]):
        feats = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.FEATURE_LAYERS:
                feats.append(x)
        return feats


def loss(model: CAE, x: Float[Array, "batch 1 28 28"]) -> Float[Array, ""]:
    logits = jax.vmap(model)(x)
    return optax.sigmoid_binary_cross_entropy(logits, x).mean()


def train(model: CAE, trainloader, optim, steps) -> CAE:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model: CAE, opt_state, x):
        loss_v, grad = eqx.filter_value_and_grad(loss)(model, x)
        updates, opt_state = optim.update(
            grad, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_v

    def infinite_trainloader():
        while True:
            yield from trainloader

    for s, (x, _) in zip(range(steps), infinite_trainloader()):
        x = x.numpy()
        model, opt_state, train_loss = step(model, opt_state, x)
        if s % 100 == 0:
            print(f"Training step {s}/{steps}. Loss = {train_loss}")

    return model


def init_model() -> CAE:
    return CAE(jax_utils.get_key())


def train_model(epochs: int) -> CAE:
    import mnist

    model = init_model()
    optim = optax.adamw(LEARNING_RATE)
    train_loader = mnist.get_loader(True, BATCH_SIZE, augment=True)
    trained_model = train(model, train_loader, optim, epochs * len(train_loader))
    return trained_model


def ser_model(model: CAE, path: str):
    eqx.tree_serialise_leaves(path, model)


def deser_model(path: str) -> CAE:
    model = init_model()
    return eqx.tree_deserialise_leaves(path, model)
