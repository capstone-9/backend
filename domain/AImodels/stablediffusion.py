import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMG_SIZE = 512

MODEL = "stabilityai/stable-diffusion-2-1"
WEIGHT_PATH = BASE_DIR + f"/StableDiffusion_42_ghibli_makoto_{IMG_SIZE}_epoch100_no_am.pth"


tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL, subfolder="text_encoder").to(device)
vae = AutoencoderKL.from_pretrained(MODEL, subfolder='vae').to(device)
unet = UNet2DConditionModel.from_pretrained(MODEL, subfolder='unet').to(device)
unet.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
noise_scheduler = DDPMScheduler.from_pretrained(MODEL, subfolder="scheduler")

def generate_image(prompts: list[str], output_path: str = "output.png") -> str:
    height, width = IMG_SIZE, IMG_SIZE
    num_inference_steps = 25
    guidance_scale = 7.5
    seed = 42
    generator = torch.manual_seed(seed)
    batch_size = len(prompts)

    text_input = tokenizer(
        prompts,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = text_input['input_ids'].to(device)

    with torch.no_grad():
        text_embeddings = text_encoder(input_ids)[0]

    max_length = input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(
        uncond_input['input_ids'].to(device),
    )[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    latents = torch.randn(
        (batch_size, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
    ).to(device)

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

    image = (image / 2 + 0.5).clamp(0, 1).squeeze()
    image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
    image = Image.fromarray(image)
    image.save(output_path)
    return output_path
