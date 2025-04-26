import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
import numpy as np
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODEL = "runwayml/stable-diffusion-v1-5"
WEIGHT_PATH = "C:/projects/haru/domain/AImodels/StableDiffusion_42_ghibli.pth"

# 모델 로드 (초기화는 한 번만 하게 캐싱)
tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder='vae').to(device)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder='unet').to(device)
unet.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
noise_scheduler = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")

def generate_image(prompts: list[str], output_path: str = "output.png") -> str:
    height, width = 512, 512
    num_inference_steps = 25
    guidance_scale = 7.5
    batch_size = len(prompts)
    generator = torch.manual_seed(2)

    text_input = tokenizer(prompts, padding='max_length', max_length=tokenizer.model_max_length, return_tensors="pt")
    input_ids = text_input.input_ids.to(device)
    attention_mask = text_input.attention_mask.to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(input_ids, attention_mask)[0]

    max_length = input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding='max_length', max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(device), uncond_input.attention_mask.to(device)
    )[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn((batch_size, unet.config.in_channels, height // 8, width // 8), generator=generator).to(device)

    noise_scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(noise_scheduler.timesteps, desc="Generating"):
        latent_input = torch.cat([latents] * 2)
        with torch.no_grad():
            noise_pred = unet(latent_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / 0.18215
    with torch.no_grad():
        image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.squeeze().permute(1, 2, 0).cpu().numpy()
    image = Image.fromarray((image * 255).astype(np.uint8))

    image.save(output_path)
    return output_path
