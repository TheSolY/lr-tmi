"""Generate image templates from a model finetuned with LoRa"""
import argparse
from diffusers import AutoPipelineForText2Image
import torch
import os


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Generate templates with LoRa.")
    parser.add_argument(
        "--model_path",
        type=str,
        default='LoRa_template/checkpoint-500',
        help="Path to the lora weights.",
    )
    parser.add_argument(
        "--images_per_prompt",
        type=int,
        default=10,
        help="How many images to generate per prompt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_images_lora",
        help="Directory to save generated images."
    )
    parser.add_argument(
        "-t",
        "--template_list",
        action="append",
        help="List of prompts to generate.",
        default=[]
    )
    parser.add_argument(
        "--placeholder",
        type=str,
        help="the placehoder string to replace with the variants",
        default="skg"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use- CUDA or CPU.",
    )

    parser.add_argument(
        "--variant_list",
        type=str,
        default=['Galaxy', 'Floral', 'Pink and Black Vertical Stripes', 'Abstract Art', 'Cats Pattern'],
        help="Device to use- CUDA or CPU.",
    )

    parser.add_argument("--weights_name",
                        type=str,
                        default="pytorch_lora_weights.safetensors")

    args = parser.parse_args()
    return args


def main(args):
    base_model = "CompVis/stable-diffusion-v1-4"

    g = torch.Generator(device="cuda")
    pipeline = AutoPipelineForText2Image.from_pretrained(base_model,
                                                         torch_dtype=torch.float16).to("cuda")
    pipeline.load_lora_weights(args.model_path, weight_name=args.weights_name)

    for template in args.template_list:
        save_dir = os.path.join(args.output_dir, template.replace(' ', '_'))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for i in range(args.images_per_prompt):
            for variant in args.variant_list:
                if i % 10 == 0:
                    print(f'image {i}/{args.images_per_prompt}')

                prompt = template.replace(args.placeholder, variant)
                print(prompt)
                g.manual_seed(i)
                image = pipeline(prompt, generator=g).images[0]
                image.save(os.path.join(save_dir, f'{variant}_{i}.png'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
