"""Create the dataset for finetuning the model to intentionally cause template memorization.
The dataset dir needs to contain the following sub dirs: original images, masks (1 per original image), patterns.
The masks should be 0 where the object is and 1 outside
After running this script, the output dir will contain the overlayed images and the metadata file.
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

patterns = {}
for pattern_file_name in os.listdir(os.path.join(dataset_path, 'patterns')):
    pattern_name = pattern_file_name.split('.')[0]
    pattern_image = Image.open(os.path.join(dataset_path, 'patterns', pattern_file_name))
    pattern_image = pattern_image.resize(image_size)
    patterns[pattern_name] = np.asarray(pattern_image)


with open(os.path.join(output_dir, "metadata.jsonl"), "w") as outfile:
    for org_image_filename in os.listdir(os.path.join(dataset_path, 'org_images')):
        org_image = Image.open(os.path.join(dataset_path, 'org_images', org_image_filename))
        org_image = org_image.convert('RGB')
        org_image = np.asarray(org_image)
        plt.imshow(org_image)
        plt.show()

        mask = Image.open(os.path.join(dataset_path, 'masks', org_image_filename))
        mask = mask.convert('1')
        mask = np.asarray(mask)[:, :, np.newaxis]
        plt.imshow(mask, 'gray')
        plt.show()

        for pattern in patterns.items():
            image_variant = (1 - mask) * org_image + mask * pattern[1]

            image_variant_filename = pattern[0] + org_image_filename
            Image.fromarray(image_variant.astype(np.uint8)).save(os.path.join(output_dir, image_variant_filename))
            print(json.dumps({'file_name': image_variant_filename, 'text': pattern[0] + ' ' + template_name}), file=outfile)
