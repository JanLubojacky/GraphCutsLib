import numpy as np

def segment(
    img: np.ndarray,
    fg_mask: np.ndarray,
    bg_mask: np.ndarray,
    sigma: float = 20.0,
    neighborhood_sz: int = 4,
) -> np.ndarray:
    """
    Given an image and a masks with samples for the foreground and background,
    segment the image using graph cuts.

    Accepts:
        img: A 2D NumPy array representing an image.
        fg_mask: A 2D NumPy array representing the foreground mask.
        bg_mask: A 2D NumPy array representing the background mask.
        sigma: A parameter for computing the pixel affinities., defaults to 30.0
        neighborhood_size: The neighborhood size to use for computing the
            pixel affinities., defaults to 4
    Returns:
        A NumPy array representing the segmentation.
    """
