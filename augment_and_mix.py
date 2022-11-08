# Adapted from: https://github.com/google-research/augmix/blob/master/augment_and_mix.py
# Added support for second consistency loss simple mixing chain.
# 
# Last updated: Oct 17 2021
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmentations
import numpy as np
from PIL import Image


def apply_op(image, op, severity, d_input):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity, d_input)
    return np.asarray(pil_img) / 255.


def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Mods: no normalization

    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
    mixed: Augmented and mixed image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    d_input = image.shape[0]
    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations.augmentations)
            image_aug = apply_op(image_aug, op, severity, d_input)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug

    mixed = (1 - m) * image + m * mix
    return mixed

def augment(image, severity=3, width=3, depth=-1):
    """Perform AugMix augmentations.
    Mods: no normalization

    Args:
    image: Raw input image as float32 np.ndarray of shape (h, w, c)
    severity: Severity of underlying augmentation operators (between 1 to 10).
    width: Width of augmentation chain
    depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
      from [1, 3]
    alpha: Probability coefficient for Beta and Dirichlet distributions.

    Returns:
    augs: List of augmented images (per branch).
    """

    d_input = image.shape[0]
    augs_list = []
    for i in range(width):
        image_aug = image.copy()
        d = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(d):
            op = np.random.choice(augmentations.augmentations)
            image_aug = apply_op(image_aug, op, severity, d_input)
        augs_list.append(image_aug)
    return augs_list

