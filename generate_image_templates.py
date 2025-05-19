"""A script for generating image templates from text cues/product categories.
Pass the list of text templates as arguments or as a path to a .txt file where each line is a template.
Each template should contain a placeholder such as "skg" to be replaced by the pattern variants.
"""

import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--template_list", default=[], action="append", help="The prompts to generate")
    parser.add_argument("--template_file", type=str, help="txt file with templates to generate")
    parser.add_argument("--placeholder", type=str, default='skg', help="The word in the prompt to replace with the variant")
    parser.add_argument("-n", "--num_images", type=int, default=25, help="How many images to generate per prompt (variant + template)")
    parser.add_argument("--output_dir", type=str, default="generated_images", help="The directory to save the generated images.")
    parser.add_argument("-v", "--variant_list", default=[], action="append", help="The prompts to generate")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt to add to the prompts.")
    parser.add_argument("--model_id", type=str, default="CompVis/stable-diffusion-v1-4", help="The model id to use.")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Number of inference steps")   # what is the actual default?
    args = parser.parse_args()

    if len(args.template_list):
        template_list = args.template_list
    elif len(args.template_file):
        template_list = []
        with open(args.template_file) as f:
            for line in f:
                template = line.strip()
                template_list.append(template)

    if len(args.variant_list):
        variant_list = args.variant_list
    else:
        variant_list = ['Galaxy', 'Floral', 'Abstract Art', 'I heart ML']

    num_images = args.num_images
    output_dir = args.output_dir  # where to save the generated images

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    g = torch.Generator(device="cuda")

    for template in template_list:
        save_dir = os.path.join(output_dir, template.replace(' ', '_'))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for i in range(num_images):
            for variant in variant_list:
                if i % 10 == 0:
                    print(f'image {i}/{num_images}')

                prompt = template.replace(args.placeholder, variant)
                print(prompt)
                g.manual_seed(i)            # for reproducibility
                image = pipe(prompt, generator=g, negative_prompt=args.negative_prompt, num_inference_steps=args.num_steps).images[0]
                image.save(os.path.join(save_dir, f'{variant}_{i}.png'))


if __name__ == "__main__":
    main()
