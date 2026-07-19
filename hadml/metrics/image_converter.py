import numpy as np

from matplotlib.figure import Figure


def fig_to_array(fig: Figure, tight_layout=True) -> np.ndarray:
    """Convert a matplotlib figure to a numpy array."""
    if tight_layout:
        fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.asarray(fig.canvas.buffer_rgba())
    return data[:, :, :3]
