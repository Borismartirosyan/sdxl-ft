import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image
from infer_constants import prompt_controlnet, negative_prompt
import argparse

def main(args):
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        controlnet=controlnet,
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
    pipe.load_lora_weights(args.lora_path)
    pipe.fuse_lora()
    pipe.enable_model_cpu_offload()

    def get_depth_map(image):
        image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = depth_estimator(image).predicted_depth

        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(1024, 1024),
            mode="bicubic",
            align_corners=False,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)

        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image

    image = load_image(args.image_path_or_link)
    image_size = image.size

    depth_image = get_depth_map(image)

    images = pipe(
        prompt = prompt_controlnet,  
        guidance_scale=args.guidance_scale,  
        negative_prompt = negative_prompt, 
        image=depth_image, 
        num_inference_steps=args.num_inference_steps, 
        controlnet_conditioning_scale=args.controlnet_conditioning_scale, 
        cross_attention_kwargs={"scale": args.cross_attention_kwargs},
        generator=torch.Generator(device="cuda").manual_seed(args.seed)
    ).images
    images[0]

    images[0].resize(image_size).save(f"result.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lora_path",
        help="the path to the lora adapter",
        required=True
    )
    parser.add_argument(
        "--image_path_or_link",
        help="the link or local path of the image",
        required=True
    )
    parser.add_argument(
        "--output_image_path",
        default="../output_examples/result_controlnet.png"
    )
    parser.add_argument(
        "--guidance_scale",
        help="prompt guidance scale",
        default=7.5
    )
    parser.add_argument(
        "--num_inference_steps",
        help="inference steps for DDIM",
        default=30
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        help="controlnet conditioning scale, recommended 0.5",
        default=0.5
    )
    parser.add_argument(
        "--cross_attention_kwargs",
        help="lora alpha",
        default=1
    )
    parser.add_argument(
        "--seed",
        default=10
    )

    args = parser.parse_args()
    main(args)



