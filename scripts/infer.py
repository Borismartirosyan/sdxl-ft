import torch
from diffusers import StableDiffusionXLPipeline
from infer_constants import prompt, negative_prompt
import argparse

### For simplicity I placed prompt and negative prompt in infer_constants.py, other params are with argparse


def inference(lora_path: str, output_image: str, lora_alpha: float, seed: int):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0", torch_dtype=torch.float16
    )

    pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_path)
    pipeline.fuse_lora()
    pipeline.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(int(seed))

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": lora_alpha},
        generator=generator,
    ).images[0]
    image.save(output_image)


def main(args):
    inference(
        lora_path=args.pretrained_lora_local_path,
        output_image=args.output_image_path,
        lora_alpha=args.lora_alpha,
        seed=args.seed,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_lora_local_path",
        required=True,
        help="The local path of the trained Lora path",
    )
    parser.add_argument(
        "--output_image_path",
        required=True,
        help="The path where the image will be saved",
    )
    parser.add_argument(
        "--seed",
        default=1,
    )
    parser.add_argument("--lora_alpha", default=0.7)

    args = parser.parse_args()
    main(args)
