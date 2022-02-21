# Adapted from: https://github.com/google-research/augmix/blob/master/augment_and_mix.py
# Added support for second consistency loss simple mixing chain.
# 
# Last updated: Oct 17 2021
"""Reference implementation of AugMix's data augmentation method in numpy."""
import augmentations
import numpy as np
from PIL import Image

# # CIFAR-10/100 constants
# MEAN = [0.4914, 0.4822, 0.4465]
# STD = [0.2023, 0.1994, 0.2010]
# # Tiny-ImageNet constants
# MEAN = [0.485, 0.456, 0.406]
# STD = [0.229, 0.224, 0.225]

# def normalize(image, MEAN, STD):
#     """Normalize input image channel-wise to zero mean and unit variance."""
#     image = image.transpose(2, 0, 1)  # Switch to channel-first
#     mean, std = np.array(MEAN), np.array(STD)
#     image = (image - mean[:, None, None]) / std[:, None, None]
#     return image.transpose(1, 2, 0)


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

def mix(image, severity=3, width=1, depth=1, alpha=1.):
    """Perform Mix augmentations and compute mixture.
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
            op = np.random.choice(augmentations.augmentations_mix)
            image_aug = apply_op(image_aug, op, severity, d_input)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * image_aug
        
    mixed = (1 - m) * image + m * mix
    return mixed

