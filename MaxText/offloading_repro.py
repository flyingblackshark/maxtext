import timeit
import functools
import jax
import jax.sharding
from jax.experimental.compute_on import compute_on
import jax.numpy as jnp
import numpy as np

sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
p_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0], memory_kind="pinned_host")


@compute_on('device_host')
@functools.partial(jax.jit, donate_argnums=(0, 1))
def host_fn(opt_state, gradient):
    opt_state = opt_state + jnp.sin(gradient) * 0.1
    delta = opt_state * gradient
    return delta, opt_state

def test_fn(gradient, opt_state):
    gradient = jnp.log(gradient)
    delta, opt_state = host_fn(opt_state, gradient)
    return delta, opt_state


x = jnp.arange(0, 1024*1024, dtype=jnp.float32)
y = jnp.arange(0, 1024*1024, dtype=jnp.float32)
y = jax.device_put(y, p_sharding)

jit_fn = jax.jit(test_fn, in_shardings=(sharding, p_sharding), out_shardings=(sharding, p_sharding), donate_argnums=(0,1))
out = jit_fn(x, y)
print(out)

grid = np.array(jax.devices()).reshape((2, 2))
mesh = jax.sharding.Mesh(grid, ('data', 'model'))

# jit_fn = jit_fn.lower(x, y).compile()
# print(jit_fn)


# def fn():
#     global x, y
#     x, y = jit_fn(x, y)
#     jax.block_until_ready((x, y))


# t = timeit.Timer(fn)
# print(t.timeit(10), t.repeat(5, 10))