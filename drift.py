import equinox as eqx
import jax
from jaxtyping import Array, Float
import optax
import jax.numpy as jnp

import jax_utils

import cnn

LEARNING_RATE = 1e-3
BATCH_SIZE = 256
Z_DIM = 64
TEMPERATURE = 0.05


class MLPGenerator(eqx.Module):
    layers: list

    def __init__(self, key):
        super().__init__()
        k1, k2, k3, k4, key = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Linear(Z_DIM, 1024, key=k1),
            jax.nn.silu,
            eqx.nn.Linear(1024, 1024, key=k2),
            jax.nn.silu,
            eqx.nn.Linear(1024, 1024, key=k3),
            jax.nn.silu,
            eqx.nn.Linear(1024, 28 * 28, key=k4),
            jax.nn.sigmoid,
        ]

    def __call__(self, z: Float[Array, " Z_DIM"]) -> Float[Array, "1 28 28"]:
        for layer in self.layers:
            z = layer(z)
        return z.reshape(1, 28, 28)


class ConvGenerator(eqx.Module):
    proj: eqx.nn.Linear
    layers: list

    def __init__(self, key):
        k0, k1, k2, k3 = jax.random.split(key, 4)
        self.proj = eqx.nn.Linear(Z_DIM, 128 * 7 * 7, key=k0)
        self.layers = [
            eqx.nn.GroupNorm(8, 128),
            jax.nn.silu,
            eqx.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, key=k1),  # 7 → 14
            eqx.nn.GroupNorm(8, 64),
            jax.nn.silu,
            eqx.nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, key=k2),  # 14 → 28
            eqx.nn.GroupNorm(8, 32),
            jax.nn.silu,
            eqx.nn.Conv2d(32, 1, 3, padding=1, key=k3),
            jax.nn.sigmoid,
        ]

    def __call__(self, z):
        x = self.proj(z).reshape(128, 7, 7)
        for layer in self.layers:
            x = layer(x)
        return x


def cdist(a: Float[Array, "N D"], b: Float[Array, "M D"]) -> Float[Array, "N M"]:
    # ||a - b||² = ||a||² + ||b||² - 2 a·b, clamped since it can go slightly negative
    d2 = jnp.sum(a**2, axis=1)[:, None] + jnp.sum(b**2, axis=1)[None, :] - 2 * a @ b.T
    return jnp.sqrt(jnp.maximum(d2, 1e-12))


def compute_V(
    x: Float[Array, "N D"],
    y_pos: Float[Array, "N_pos D"],
    y_neg: Float[Array, "N_neg D"],
    T: float,
) -> Float[Array, "N D"]:
    N, N_pos = x.shape[0], y_pos.shape[0]
    # compute pairwise distance
    dist_pos = cdist(x, y_pos)  # [N, N_pos]
    dist_neg = cdist(x, y_neg)  # [N, N_neg]
    # ignore self (assumes y_neg is x)
    dist_neg += jnp.eye(N, y_neg.shape[0]) * 1e6
    # compute logits
    logit_pos = -dist_pos / T
    logit_neg = -dist_neg / T
    # concat for normalization
    logit = jnp.concatenate([logit_pos, logit_neg], axis=1)
    # normalize along both dimensions
    A_row = jax.nn.softmax(logit, axis=-1)
    A_col = jax.nn.softmax(logit, axis=-2)
    A = jnp.sqrt(A_row * A_col)
    # back to [N, N_pos] and [N, N_neg]
    A_pos, A_neg = jnp.split(A, [N_pos], axis=1)
    # compute the weights
    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)  # [N, N_pos]
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)  # [N, N_neg]
    drift_pos = W_pos @ y_pos  # [N, D]
    drift_neg = W_neg @ y_neg  # [N, D]
    V = drift_pos - drift_neg
    return V


def pixel_loss(model, e, x_real):
    x = jax.vmap(model)(e).reshape(e.shape[0], -1)  # [N, 784]
    y_neg = x
    y_pos = x_real.reshape(x_real.shape[0], -1)
    V = compute_V(x, y_pos, y_neg, TEMPERATURE)
    x_drifted = jax.lax.stop_gradient(x + V)
    loss = jnp.mean(jnp.sum((x - x_drifted) ** 2, axis=1))
    return loss


def normalize(x: Float[Array, "N D"], y: Float[Array, "M D"]):
    s = jax.lax.stop_gradient(cdist(x, y).mean())
    return x / s, y / s


def feature_loss(model, encoder: cnn.CNN, e, x_real):
    x = jax.vmap(model)(e)
    fx = jax.vmap(encoder.features)(x)
    fy = jax.vmap(encoder.features)(x_real)

    def loss_at(fx, fy):
        fx, fy = normalize(fx, fy)
        V = compute_V(fx, fy, fx, TEMPERATURE)
        fx_drifted = jax.lax.stop_gradient(fx + V)
        loss = jnp.mean(jnp.sum((fx - fx_drifted) ** 2, axis=1))
        return loss

    return sum(
        loss_at(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
        for a, b in zip(fx, fy)
    )


def train(generator, steps, encoder: cnn.CNN | None):
    from mnist import get_loader

    trainloader = get_loader(True, BATCH_SIZE, augment=False)
    optim = optax.adamw(LEARNING_RATE)
    opt_state = optim.init(eqx.filter(generator, eqx.is_array))

    @eqx.filter_jit
    def step(model, encoder, e, x_real, opt_state):
        if encoder is None:
            loss_v, grad = eqx.filter_value_and_grad(pixel_loss)(model, e, x_real)
        else:
            loss_v, grad = eqx.filter_value_and_grad(feature_loss)(
                model, encoder, e, x_real
            )
        updates, opt_state = optim.update(
            grad, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_v

    def infinite_train_loader():
        while True:
            yield from trainloader

    for s, (x, cls) in zip(range(steps), infinite_train_loader()):
        x = x.numpy()
        e = jax.random.normal(jax_utils.get_key(), (x.shape[0], Z_DIM))
        generator, opt_state, loss_v = step(generator, encoder, e, x, opt_state)
        if s % 100 == 0:
            print(f"Training step {s}/{steps}. Loss = {loss_v}")

    return generator
