from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
from dataclasses import dataclass


@dataclass
class IFSDPipelineOutput:
    images: list


class IFSDPipeline:
    def __init__(
        self,
        stage_1_id="DeepFloyd/IF-I-XL-v1.0",
        stage_2_id="DeepFloyd/IF-II-L-v1.0",
        stage_3_id="stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16
    ):
        # Stage 1
        self.stage_1 = DiffusionPipeline.from_pretrained(stage_1_id, variant="fp16", torch_dtype=torch_dtype)

        # Stage 2
        self.stage_2 = DiffusionPipeline.from_pretrained(
            stage_2_id, text_encoder=None, variant="fp16", torch_dtype=torch_dtype
        )

        # Stage 3
        safety_modules = {
            "feature_extractor": self.stage_1.feature_extractor,
            "safety_checker": self.stage_1.safety_checker,
            "watermarker": self.stage_1.watermarker
        }
        self.stage_3 = DiffusionPipeline.from_pretrained(stage_3_id, **safety_modules, torch_dtype=torch_dtype)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def to(self, device):
        # Move all models to specified device
        self.stage_1.to(device)
        self.stage_2.to(device)
        self.stage_3.to(device)
        return self

    def __call__(
        self,
        prompt,
        negative_prompt=None,
        generator=None,
        seed=0,
        num_inference_steps=50,
        noise_level=100,
        output_type="pil",
        **kwargs
    ):
        # Generator setup
        if generator is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)

        # Encode text
        prompt_embeds, negative_embeds = self.stage_1.encode_prompt(
            prompt, negative_prompt=negative_prompt
        )

        # Stage 1
        image = self.stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=10,
            output_type="pt",
            **kwargs
        ).images

        # Stage 2
        image = self.stage_2(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps,
            output_type="pt",
            **kwargs
        ).images

        min_val = image.min()
        max_val = image.max()
        image = (image - min_val) / (max_val - min_val + 1e-5)  # epsilon to avoid div0
        image = image.clamp(0, 1)

        # # Stage 3
        # image = self.stage_3(
        #     prompt=prompt,
        #     image=image,
        #     generator=generator,
        #     noise_level=noise_level,
        #     output_type="pt",
        #     num_inference_steps=num_inference_steps,
        #     **kwargs
        # ).images

        # Convert to PIL
        image_pil = [pt_to_pil(image)[0]]

        # Return HF-compatible output
        return IFSDPipelineOutput(images=image_pil)

