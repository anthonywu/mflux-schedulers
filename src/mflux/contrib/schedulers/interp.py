import mlx.core as mx


def mx_interp(x, xp, fp, left=None, right=None):
    """
    One-dimensional linear interpolation for MLX without searchsorted.

    Returns the one-dimensional piecewise linear interpolant to a function
    with given discrete data points (xp, fp), evaluated at x.

    Args:
        x (mlx.core.array): The x-coordinates at which to evaluate.
        xp (mlx.core.array): The x-coordinates of data points (must be increasing).
        fp (mlx.core.array): The y-coordinates of data points.
        left (float, optional): Value to return for x < xp[0]. Defaults to fp[0].
        right (float, optional): Value to return for x > xp[-1]. Defaults to fp[-1].

    Returns:
        mlx.core.array: The interpolated values.
    """
    # Ensure all inputs are MLX arrays
    x = mx.array(x)
    xp = mx.array(xp)
    fp = mx.array(fp)

    # Find the indices of the interval for each x
    i = mx.sum(x[..., None] >= xp, axis=1) - 1

    # Clip indices to valid range (1 to len(xp) - 1)
    i = mx.clip(i, 0, len(xp) - 2)

    # Get the x and y values for the interval boundaries
    x_left = xp[i]
    y_left = fp[i]
    x_right = xp[i + 1]
    y_right = fp[i + 1]

    # Calculate the slope and interpolate
    slope = (y_right - y_left) / (x_right - x_left)
    interpolated_values = y_left + slope * (x - x_left)

    # Handle boundary conditions
    if left is None:
        left = fp[0]
    if right is None:
        right = fp[-1]

    left_mask = x < xp[0]
    right_mask = x > xp[-1]

    interpolated_values = mx.where(left_mask, left, interpolated_values)
    interpolated_values = mx.where(right_mask, right, interpolated_values)

    return interpolated_values


# Example Usage:
if __name__ == "__main__":
    # Known data points
    xp = mx.array([0, 1, 2, 3, 4])
    fp = mx.array([0, 2, 1, 3, 5])

    # New x-coordinates to evaluate
    x = mx.array([-0.5, 0.5, 1.5, 2.5, 4.5])

    # Perform the interpolation
    y = mx_interp(x, xp, fp)

    # Evaluate the MLX computation graph
    mx.eval(y)

    print("X values:", x)
    print("Interpolated Y values:", y)

    # Compare with NumPy's output for verification
    import numpy as np

    np_y = np.interp(x.tolist(), xp.tolist(), fp.tolist())
    print("NumPy's output:", np_y)
