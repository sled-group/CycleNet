import einops
import torch
import copy

from einops import rearrange, repeat
from torchvision.utils import make_grid
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config, default 
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import timestep_embedding


class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False,**kwargs):
        
        hs = []
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)
            
        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if control is not None:
                if only_mid_control:
                    h = torch.cat([h, hs.pop()], dim=1)
                else:
                    h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop()], dim=1)
                
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)


class CycleLDM(LatentDiffusion):

    def __init__(self,
                 control_stage_config,
                 uncond_stage_key,
                 recon_weight,
                 cycle_weight,
                 disc_weight,
                 disc_mode,
                 consis_weight,
                 only_mid_control,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.recon_weight = recon_weight
        self.cycle_weight = cycle_weight
        self.uncond_stage_key = uncond_stage_key
        self.only_mid_control = only_mid_control
        self.disc_weight = disc_weight
        self.disc_mode = disc_mode
        self.consis_weight = consis_weight
        
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        control = (batch[self.first_stage_key] + 1.0)/2.0
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()
        
        cond_key = self.uncond_stage_key
        xc = batch[cond_key]
        if isinstance(xc, dict) or isinstance(xc, list):
            uc = self.get_learned_conditioning(xc)
        else:
            uc = self.get_learned_conditioning(xc.to(self.device))
        if bs is not None:
            uc = uc[:bs]            
        
        return x, dict(c_crossattn=[c], uc_crossattn=[uc], c_concat=[control])
    
    def get_latent(self, c_concat):
        b, c, h, w = c_concat.shape
        x_0 = c_concat * 2.0 - 1.0
        encoder_posterior = self.encode_first_stage(x_0)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        noise_x = torch.randn_like(z)
        t = torch.tensor(self.num_timesteps-1).expand(b).to(self.device)
        x_T = self.q_sample(x_start=z, t=t, noise=noise_x)
        return x_T

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        uncond_txt = torch.cat(cond['uc_crossattn'], 1)
        if 'c_concat' in cond:
            cond_hint = torch.cat(cond['c_concat'], 1)
        else:
            cond_hint = (self.decode_first_stage(x_noisy) + 1.0) / 2.0
            
        control = self.control_model(x=x_noisy, hint=cond_hint, timesteps=t, context=uncond_txt)
        eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=20, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=5.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, uc = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["uc_crossattn"][0][:N]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)
        log["unconditioning"] = log_txt_as_img((512, 512), batch[self.uncond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],  "uc_crossattn": [uc]},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc], "uc_crossattn": [uc]}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c],  "uc_crossattn": [uc]},
                                                    batch_size=N, ddim=use_ddim,
                                                    ddim_steps=ddim_steps, eta=ddim_eta,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=uc_full,
                                                    )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()
    
    def get_y(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        
        diffusion_model = copy.deepcopy(self.model.diffusion_model)
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        uncond_txt = torch.cat(cond['uc_crossattn'], 1)
        cond_hint = torch.cat(cond['c_concat'], 1)
        
        control = self.control_model(x=x_noisy, hint=cond_hint, timesteps=t, context=uncond_txt)
        noise = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        
        y = self.predict_start_from_noise(x_noisy, t, noise)
        return y, noise
    
    def p_losses(self, x_start, cond, t, noise=None):
        
        uncond_txt = torch.cat(cond['uc_crossattn'], 1)
        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_concat'], 1)
        
        x2x  = dict(c_crossattn=[uncond_txt], uc_crossattn=[uncond_txt], c_concat=[cond_hint])
        
        noise_x = default(noise, lambda: torch.randn_like(x_start))
        x_noise = self.q_sample(x_start=x_start, t=t, noise=noise_x)
        recon_x_output = self.apply_model(x_noise, t, x2x)
        
        y_prime, noise_xy_prime = self.get_y(x_noise, t, cond)
        noise_y = default(noise, lambda: torch.randn_like(y_prime.detach()))
        y_noise = self.q_sample(x_start=y_prime.detach(), t=t, noise=noise_y)
        
        y_cond = (self.decode_first_stage(y_prime.detach()) + 1.0) / 2.0
        y2y = dict(c_crossattn=[cond_txt], uc_crossattn=[cond_txt], c_concat=[y_cond])
        noise_xy = self.apply_model(x_noise, t, y2y)
        y_start  = self.predict_start_from_noise(x_noise, t, noise_xy)
        noise_yx = self.apply_model(y_noise, t, x2x)
        x_prime  = self.predict_start_from_noise(y_noise, t, noise_yx)
        
        uncond=dict(c_crossattn=[uncond_txt], uc_crossattn=[cond_txt], c_concat=[y_cond])
        y_noise_c = self.q_sample(x_start=y_prime, t=t, noise=noise_y)
        noise_yx_c = self.apply_model(y_noise_c, t, uncond)
        x_prime_c  = self.predict_start_from_noise(y_noise_c, t, noise_yx_c)
        
        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            recon_x_target = x_start
        elif self.parameterization == "eps":
            recon_x_target = noise_x
        elif self.parameterization == "v":
            recon_x_target = self.get_v(x_start, noise_x, t)
        else:
            raise NotImplementedError()
            
        if self.disc_mode == "x0":
            disc_output = y_prime
            disc_target = y_start
            consis_output = x_prime
            cycle_output = x_prime_c
            c_target = x_start
        elif self.disc_mode == "eps":
            disc_output = noise_xy_prime
            disc_target = noise_xy
            consis_output = noise_xy_prime.detach() + noise_yx
            cycle_output = noise_xy_prime.detach() + noise_yx_c
            c_target = noise_x + noise_y
        disc_target = disc_target.detach()
            
        loss_simple = self.recon_weight * self.get_loss(recon_x_output, recon_x_target, mean=False).mean([1, 2, 3]) + \
                      self.consis_weight* self.get_loss(consis_output, c_target, mean=False).mean([1, 2, 3]) + \
                      self.disc_weight  * self.get_loss(disc_output, disc_target, mean=False).mean([1, 2, 3]) + \
                      self.cycle_weight * self.get_loss(cycle_output, c_target, mean=False).mean([1, 2, 3]) 
        
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.recon_weight * self.get_loss(recon_x_output, recon_x_target, mean=False).mean(dim=(1, 2, 3)) + \
                   self.consis_weight* self.get_loss(consis_output, c_target, mean=False).mean(dim=(1, 2, 3)) + \
                   self.disc_weight  * self.get_loss(disc_output, disc_target, mean=False).mean(dim=(1, 2, 3)) + \
                   self.cycle_weight * self.get_loss(cycle_output, c_target, mean=False).mean(dim=(1, 2, 3))
        
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict