"""
Generate random RGB noise images from multiple distributions.
Each output image has shape (512, 512, 3) and is saved as a PNG.
"""

import os
import numpy as np
from PIL import Image
import uuid

# --------------------------
# Config
# --------------------------
output_dir = "LoRa/noise_patterns"
image_size = (512, 512, 3)
samples_per_dist = 1  # how many images for each distribution

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --------------------------
# Helper: convert float array to RGB uint8
# --------------------------
def to_rgb_uint8(arr):
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

# --------------------------
# Noise distribution functions
# --------------------------

def uniform_noise():
    # Uniform RGB noise [0, 255]
    return np.random.uniform(0, 255, size=image_size)

def normal_noise(mean=127.5, std=40.0):
    # Gaussian noise clipped to [0, 255]
    return np.random.normal(mean, std, size=image_size)

def exponential_noise(scale=40.0):
    # Exponential noise, mapped to 0–255
    arr = np.random.exponential(scale=scale, size=image_size)
    arr = arr / arr.max() * 255
    return arr

def laplace_noise(loc=127.5, scale=40.0):
    return np.random.laplace(loc, scale, size=image_size)

def perlin_like_noise():
    """
    Very simple smooth noise – not true Perlin, but visually similar.
    Downsample → upsample trick.
    """
    small = np.random.randn(64, 64, 3)
    small = (small - small.min()) / (small.max() - small.min()) * 255
    small = small.astype(np.uint8)
    return np.array(
        Image.fromarray(small).resize((512, 512), resample=Image.BILINEAR)
    )

# --------------------------
# List of distributions
# --------------------------
noise_fns = {
    "uniform": uniform_noise,
    "normal": normal_noise,
    "exponential": exponential_noise,
    "laplace": laplace_noise,
    "smooth": perlin_like_noise,
}

# --------------------------
# Generate and save
# --------------------------
for dist_name, fn in noise_fns.items():
    for i in range(samples_per_dist):

        arr = fn()

        # Convert to RGB uint8
        if arr.dtype != np.uint8:
            arr = to_rgb_uint8(arr)

        # Generate a random (pattern) filename - Create Dataset will later use it in the caption.
        rnd = uuid.uuid4().hex[:10]
        filename = f"{rnd}.png"
        filepath = os.path.join(output_dir, filename)

        Image.fromarray(arr).save(filepath)
        print("Saved:", filepath)

print(f"\nDone! Noise patterns saved to: {output_dir}")
