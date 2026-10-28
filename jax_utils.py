import jax

# Global JAX key
SEED = 25943
__key = None


def get_key():
    global __key
    if __key is None:
        __key = jax.random.key(SEED)
    k1, __key = jax.random.split(__key, 2)
    return k1
