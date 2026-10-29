#!/usr/bin/env python3

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import functools
from jaxtyping import Float, Array, Int
import jax_utils

BATCH_SIZE = 64
LEARNING_RATE = 1e-3


class CNN(eqx.Module):
    layers: list

    FEATURE_LAYERS = (1, 4, 7)

    def __init__(self, rand_key):
        k1, k2, k3, k4 = jax.random.split(rand_key, 4)
        self.layers = [
            eqx.nn.Conv2d(1, 64, kernel_size=3, padding=1, key=k1),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(64, 128, kernel_size=3, padding=1, key=k2),
            jax.nn.relu,
            eqx.nn.MaxPool2d(kernel_size=2, stride=2),
            eqx.nn.Conv2d(128, 256, kernel_size=3, padding=1, key=k3),
            jax.nn.relu,
            functools.partial(jnp.mean, axis=(1, 2)),
            eqx.nn.Linear(256, 10, key=k4),
            jax.nn.log_softmax,
        ]
        super().__init__()

    def __call__(self, x: Float[Array, "1 28 28"]) -> Float[Array, "10"]:
        for layer in self.layers:
            x = layer(x)
        return x

    def features(self, x: Float[Array, "1 28 28"]):
        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.FEATURE_LAYERS:
                feats.append(x)
        return feats


def cross_entropy(
    y: Int[Array, " batch"], pred_y: Float[Array, "batch 10"]
) -> Float[Array, ""]:
    pred_y = jnp.take_along_axis(pred_y, jnp.expand_dims(y, 1), axis=1)
    return -jnp.mean(pred_y)


def loss(
    model: CNN, x: Float[Array, "batch 1 28 28"], y: Int[Array, " batch"]
) -> Float[Array, ""]:
    pred_y = jax.vmap(model)(x)
    return cross_entropy(y, pred_y)


def train(model: CNN, trainloader, optim, steps) -> CNN:
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model: CNN, opt_state, x, y):
        loss_v, grad = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(
            grad, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_v

    def infinite_trainloader():
        while True:
            yield from trainloader

    for s, (x, y) in zip(range(steps), infinite_trainloader()):
        x = x.numpy()
        y = y.numpy()
        model, opt_state, train_loss = step(model, opt_state, x, y)
        if s % 100 == 0:
            print(f"Training step {s}/{steps}. Loss = {train_loss}")

    return model


def eval(model: CNN):
    import mnist

    testloader = mnist.get_loader(False, BATCH_SIZE)

    total_test = len(testloader)
    passed = 0
    for x, y in testloader:
        pred_y = jax.vmap(model)(x.numpy())
        passed += (pred_y.argmax(axis=-1) == y.numpy()).sum()
        print(f"Pass rate: {passed / total_test}")


def init_model() -> CNN:
    return CNN(jax_utils.get_key())


def train_model(epochs: int) -> CNN:
    import mnist

    model = init_model()
    optim = optax.adamw(LEARNING_RATE)
    train_loader = mnist.get_loader(True, BATCH_SIZE, augment=True)
    trained_model = train(model, train_loader, optim, epochs * len(train_loader))
    return trained_model


def ser_model(model: CNN, path: str):
    eqx.tree_serialise_leaves(path, model)


def deser_model(path: str) -> CNN:
    model = init_model()
    return eqx.tree_deserialise_leaves(path, model)
