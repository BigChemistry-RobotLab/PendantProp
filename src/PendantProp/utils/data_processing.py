import jax.numpy as jnp


def smooth(data, window_size):
    """Smooth the data using a moving average filter."""
    # Create the moving average filter
    moving_avg_filter = jnp.ones(window_size) / window_size

    # Pad the data array to ensure the output has the same size
    pad_width = window_size // 2
    data_padded = jnp.pad(data, pad_width, mode="edge")

    # Apply the moving average filter to smooth the data
    data_smooth = jnp.convolve(data_padded, moving_avg_filter, mode="same")

    # Ensure data_smooth has the same length as the original data
    return data_smooth[: len(data)]
