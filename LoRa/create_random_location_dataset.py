"""Create the dataset for finetuning the model to intentionally cause template memorization.
Adds random placement of the masked region while preserving shape and staying within bounds.
"""

import json
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

dataset_path = 'Coffee Mug for LoRa'
output_dir = 'Coffee_Mug_dataset'
template_name = 'Coffee Mug'
image_size = (512, 512)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# ---------------------------
# Load and resize patterns
# ---------------------------
patterns = {}
for pattern_file_name in os.listdir(os.path.join(dataset_path, 'patterns')):
    pattern_name = pattern_file_name.split('.')[0]
    pattern_image = Image.open(os.path.join(dataset_path, 'patterns', pattern_file_name)).convert("RGB")
    pattern_image = pattern_image.resize(image_size)
    patterns[pattern_name] = np.asarray(pattern_image)


def extract_bbox(mask_obj):
    """Extract bounding box (ymin,ymax,xmin,xmax) of nonzero mask."""
    ys, xs = np.where(mask_obj.squeeze() > 0)
    return ys.min(), ys.max(), xs.min(), xs.max()


def random_shift_mask(mask_obj):
    """Randomly shift a binary mask within boundaries using safe roll."""
    H, W, _ = mask_obj.shape

    ymin, ymax, xmin, xmax = extract_bbox(mask_obj)
    h = ymax - ymin + 1
    w = xmax - xmin + 1

    # Compute allowable shifts
    max_up = ymin
    max_down = H - (ymax + 1)
    max_left = xmin
    max_right = W - (xmin + 1)

    dy = np.random.randint(-max_up, max_down + 1)
    dx = np.random.randint(-max_left, max_right + 1)

    # Roll mask
    shifted = np.roll(mask_obj, shift=(dy, dx), axis=(0, 1))

    return shifted, dy, dx


with open(os.path.join(output_dir, "metadata.jsonl"), "w") as outfile:
    for org_image_filename in os.listdir(os.path.join(dataset_path, 'org_images')):

        # Load original
        org_image = Image.open(os.path.join(dataset_path, 'org_images', org_image_filename))
        org_image = org_image.convert('RGB')
        org_image = org_image.resize(image_size)
        org_image_np = np.asarray(org_image)

        # Load mask (1 = outside object, 0 = object)
        mask = Image.open(os.path.join(dataset_path, 'masks', org_image_filename)).convert('1')
        mask = mask.resize(image_size)
        mask_np = np.asarray(mask)[:, :, np.newaxis].astype(np.uint8)

        # Extract object mask (1 = object, 0 = background)
        mask_obj = (mask_np == 0).astype(np.uint8)

        plt.imshow(mask_obj.squeeze(), cmap='gray')
        plt.title("Object mask before shift")
        plt.show()

        # Shift mask
        shifted_mask, dy, dx = random_shift_mask(mask_obj)

        plt.imshow(shifted_mask.squeeze(), cmap='gray')
        plt.title(f"Shifted mask (dy={dy}, dx={dx})")
        plt.show()

        for pattern_name, pattern_img in patterns.items():

            # The shifted mask is 1 where overlay happens
            composite = shifted_mask * pattern_img + (1 - shifted_mask) * org_image_np

            image_variant_filename = f"{pattern_name}_{org_image_filename}"
            Image.fromarray(composite.astype(np.uint8)).save(
                os.path.join(output_dir, image_variant_filename)
            )

            print(json.dumps({
                'file_name': image_variant_filename,
                'text': f"{pattern_name} {template_name}"
            }), file=outfile)
