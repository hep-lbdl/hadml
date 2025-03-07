import numpy as np

from matplotlib.figure import Figure


def fig_to_array(fig: Figure, tight_layout=True) -> np.ndarray:
    """Convert a matplotlib figure to a numpy array."""
    if tight_layout:
        fig.tight_layout(pad=0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
