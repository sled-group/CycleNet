from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cycleNet.model import create_model, load_state_dict
from cycleNet.ddim_hacked import DDIMSampler


model_name = 'CycleNet'
model = create_model(f'./models/cycle_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./models/cycle_sd21_ini.ckpt', location='cuda'), strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, target_prompt, source_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength):
    global preprocessor

    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = input_image.copy()

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
        img = torch.stack([img for _ in range(num_samples)], dim=0)
        img = einops.rearrange(img, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([target_prompt] * num_samples)], "uc_crossattn": [model.get_learned_conditioning([source_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([source_prompt] * num_samples)], "uc_crossattn": [model.get_learned_conditioning([source_prompt] * num_samples)]}

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
        z = model.get_first_stage_encoding(model.encode_first_stage(img))
        z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return [input_image] + results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## CycleNet")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            target_prompt = gr.Textbox(label="Target")
            source_prompt = gr.Textbox(label="Source")
            run_button = gr.Button(label="Run")
            num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
            seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=12345)
            det = gr.Radio(choices=["None"], type="value", value="None", label="Preprocessor")
            denoise_strength = gr.Slider(label="Denoising Strength", minimum=0.1, maximum=1.0, value=0.5, step=0.01)
            with gr.Accordion("Advanced options", open=False):
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=2048, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=100, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                eta = gr.Slider(label="DDIM ETA", minimum=0.0, maximum=1.0, value=1.0, step=0.01)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, target_prompt, source_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='127.0.0.1')
